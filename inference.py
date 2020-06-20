import pandas as pd
from model import BertBaseUncased
import torch
import numpy as np
import string
from tqdm import tqdm
import dataset
import config

TEST_FILE = 'Data/test.csv'
df_test = pd.read_csv(TEST_FILE)
sample = pd.read_csv('Data/sample_submission.csv')


test_dataset = dataset.TweetDataset( 
            tweet=df_test.text.values,
            sentiment = df_test.sentiment.values,
            selected_text = df_test.text.values # wont need this just so that the data loader works
            )

test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.VALID_BATCH_SIZE,
            shuffle = False
            )

device = torch.device('cuda')
print('Running on ',device)
model = BertBaseUncased().to(device)

model.load_state_dict(torch.load('model.bin'))

def test_fn(data_loader,model,device):
    model.eval()
    fin_output_start = []
    fin_output_end = []
    fin_padding_len = []
    fin_tweet_tokens = []
    fin_orig_sentiment = []
    fin_orig_selected = []
    fin_orig_tweet = []
    final_output_text = []

    
    with torch.no_grad():
        tk0 = tqdm(data_loader,total = len(data_loader))
        for bi,d in enumerate(tk0):
            ids = d['ids']
            token_type_ids = d['token_type_ids']
            mask = d['mask']
            tweet_tokens = d['tweet_tokens']
            orig_selected = d['orig_selected_text']
            orig_tweet = d['orig_tweet']
            orig_sentiment = d['orig_sentiment']
            padding_len = d['padding_len']
            
           
            ids = ids.to(device,dtype = torch.long)
            token_type_ids = token_type_ids.to(device,dtype = torch.long)
            mask = mask.to(device,dtype = torch.long)   
            
            o1,o2 = model(
                    ids,
                    mask,
                    token_type_ids
                    )
            
            fin_output_start.append(torch.sigmoid(o1).cpu().detach().numpy())
            fin_output_end.append(torch.sigmoid(o2).cpu().detach().numpy())        
            fin_padding_len.extend(padding_len.cpu().detach().numpy().tolist())
            
            fin_tweet_tokens.extend(tweet_tokens)
            fin_orig_sentiment.extend(orig_sentiment)
            fin_orig_selected.extend(orig_selected)
            fin_orig_tweet.extend(orig_tweet)
            
    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)
        
    threshold = 0.2   
    #print(fin_output_start,'-start')
    #print(fin_output_end,'-end')
    for j in range(len(fin_tweet_tokens)):
        tweet_tokens = fin_tweet_tokens[j]
        padding_len = fin_padding_len[j]
        original_tweet = fin_orig_tweet[j]
        sentiment = fin_orig_sentiment[j]
            
        if padding_len>0:
            mask_start = fin_output_start[j,:][:-padding_len]>=threshold 
            mask_end = fin_output_end[j,:][:-padding_len]>=threshold 
                
        else:
            mask_start = fin_output_start[j,:]>=threshold 
            mask_end = fin_output_end[j,:]>=threshold 
                 
        mask = [0]*len(mask_start)
        idx_start = np.nonzero(mask_start)[0]
        idx_end = np.nonzero(mask_end)[0]
            
        if len(idx_start)>0:
            idx_start = idx_start[0]
                
            if len(idx_end)>0:
                idx_end = idx_end[0]
            else:
                idx_end = idx_start
                    
        else:
            idx_start = 0
            idx_end = 0
                
        for mj in range(idx_start,idx_end+1):
            mask[mj] = 1
                
            
        output_tokens = [x for p,x in enumerate(tweet_tokens.split()) if mask[p]==1]
        output_tokens = [x for x in output_tokens if x not in ('[CLS]','[SEP]')]
            
        final_output = ''
            
        for ot in output_tokens:
            if ot.startswith('##'):
                         final_output = final_output+ot[2:]
            elif len(ot) == 1 and ot in string.punctuation:
                         final_output = final_output+ot
            else:
                         final_output = final_output + " " + ot
                    
        final_output = final_output.strip()
        if sentiment == 'neutral' or len(original_tweet.split())<=2:
              final_output = original_tweet
        final_output_text.append(final_output)
        
        #print(final_output,'----',target_string)
    return final_output_text

selected_texts = test_fn(test_dataloader,model, device)
sample['selected_text'] = selected_texts

sample.to_csv('submission_bert_baseline.csv')
