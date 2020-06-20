<<<<<<< HEAD
import config
import torch
import pandas as pd

class TweetDataset:
    def __init__(self,tweet,sentiment,selected_text):
        self.tweet = tweet
        self.selected_text = selected_text
        self.sentiment = sentiment
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER
        
        
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self,item):
        tweet = self.tweet[item]
        selected_text = self.selected_text[item]
        sentiment = self.sentiment[item]
        
        len_st = len(selected_text)
        idx0 = -1 # start inex
        idx1 = -1 # end index
        
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
                
                if tweet[ind:ind+len_st] == selected_text:
                    idx0 = ind # start index
                    idx1 = ind+len_st-1 # end index
                    break
                
                
        char_targets = [0]*len(tweet)
        if idx0!=-1 and idx1!=-1:
            for j in range (idx0,idx1+1):
                    char_targets[j] = 1
                    
        tok_tweet = self.tokenizer.encode(tweet)
        tweet_ids = tok_tweet.ids[1:-1]
        tok_tweet_offsets = tok_tweet.offsets[1:-1] 
        
        targets = []
        for j,(offset1,offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1:offset2])>0:
                targets.append(j)
        
        targets_start = targets[0]
        target_end = targets[-1]
        
        
        sentiment_id = {
                'positive': 3893,
                'negative':4997,
                'neutral':8699
                    }
        
        tok_tweet_ids = [101] + [sentiment_id[sentiment]] + [102] + tweet_ids + [102]
        tok_type_ids = [0,0,0] + [1]*len(tweet_ids) + [1]
        mask = [1]*len(tok_type_ids)
        tok_tweet_offsets = [(0,0)]*3 + tok_tweet_offsets + [(0,0)]
        targets_start+=3
        target_end+=3
        
        
        
        padding_len = config.MAX_LEN - len(tok_tweet_ids)
        
        if padding_len>0:
 
            ids = tok_tweet_ids+([0]*padding_len)
            mask = mask+([0]*padding_len)
            tok_type_ids = tok_type_ids+([0]*padding_len)
            offsets = tok_tweet_offsets + ([(0,0)]*padding_len)
        else:
            ids = tok_tweet_ids[:config.MAX_LEN]
            mask = mask[:config.MAX_LEN]
            tok_type_ids = tok_type_ids[:config.MAX_LEN]
            offsets = tok_tweet_offsets[:config.MAX_LEN]
            
        
            
        return {
                'ids':torch.tensor(ids,dtype = torch.long),
                'mask':torch.tensor(mask,dtype = torch.long),
                'token_type_ids':torch.tensor(tok_type_ids,dtype = torch.long),
                'targets_start':torch.tensor(targets_start,dtype = torch.long),
                'targets_end':torch.tensor(target_end,dtype = torch.long),                
                'sentiment':sentiment,
                'orig_tweet': self.tweet[item],
                'orig_sentiment':self.sentiment[item],
                'orig_selected_text':self.selected_text[item],
                'offsets':torch.tensor(offsets,dtype = torch.long)
                
                }

df = pd.read_csv(config.TRAINING_FILE)
ds = TweetDataset(df.text.values,df.sentiment.values,df.selected_text.values)
dl = torch.utils.data.DataLoader(ds,batch_size=2,shuffle = True)

d = next(iter(dl))
d


=======
import config
import torch
from model import Sentiment
import utils
import pandas as pd

class TweetDataset:
    def __init__(self,tweet,selected_text):
        self.tweet = tweet
        self.selected_text = selected_text
        self.max_len = config.MAX_LEN
        self.tokenizer = config.TOKENIZER
        
        
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self,item):
        tweet = self.tweet[item]
        selected_text = self.selected_text[item]
        
        len_st = len(selected_text)
        idx0 = -1 # start inex
        idx1 = -1 # end index
        
        for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
                
                if tweet[ind:ind+len_st] == selected_text:
                    idx0 = ind # start index
                    idx1 = ind+len_st-1 # end index
                    break
                
                
        char_targets = [0]*len(tweet)
        if idx0!=-1 and idx1!=-1:
            for j in range (idx0,idx1+1):
                    char_targets[j] = 1
                    
        tok_tweet = self.tokenizer.encode(tweet)
        tweet_ids = tok_tweet.ids[1:-1]
        tok_tweet_offsets = tok_tweet.offsets[1:-1] 
        
        targets = []
        for j,(offset1,offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1:offset2])>0:
                targets.append(j)
        
        targets_start = targets[0]
        target_end = targets[-1]
        
        
        # predicting sentiment
        device = 'cuda'
        model = Sentiment().to(device)
        model.load_state_dict(torch.load(config.SENTIMENT_MODEL_PATH))
        
        tweet_enc = utils.preprocess_tweet(tweet,config.Plrty_Tokenizer,config.MAX_LEN,device)
        
        #print(tweet_enc['ids'],tweet_enc['ids'].shape)
        out = model(tweet_enc['ids'].view(1,-1),tweet_enc['mask'].view(1,-1),tweet_enc['type_ids'].view(1,-1))
         
        _,pred = torch.max(torch.softmax(out,dim=1),dim=1)
        
        
        sentiment = 'negative'
        if pred==1:
            sentiment = 'positive'
        
        
        sentiment_id = {
                'positive': 3893,
                'negative':4997                
                    }
        
        tok_tweet_ids = [101] + [sentiment_id[sentiment]] + [102] + tweet_ids + [102]
        tok_type_ids = [0,0,0] + [1]*len(tweet_ids) + [1]
        mask = [1]*len(tok_type_ids)
        tok_tweet_offsets = [(0,0)]*3 + tok_tweet_offsets + [(0,0)]
        targets_start+=3
        target_end+=3
        
        
        
        padding_len = config.MAX_LEN - len(tok_tweet_ids)
        
        if padding_len>0:
 
            ids = tok_tweet_ids+([0]*padding_len)
            mask = mask+([0]*padding_len)
            tok_type_ids = tok_type_ids+([0]*padding_len)
            offsets = tok_tweet_offsets + ([(0,0)]*padding_len)
        else:
            ids = tok_tweet_ids[:config.MAX_LEN]
            mask = mask[:config.MAX_LEN]
            tok_type_ids = tok_type_ids[:config.MAX_LEN]
            offsets = tok_tweet_offsets[:config.MAX_LEN]
            
        
            
        return {
                'ids':torch.tensor(ids,dtype = torch.long),
                'mask':torch.tensor(mask,dtype = torch.long),
                'token_type_ids':torch.tensor(tok_type_ids,dtype = torch.long),
                'targets_start':torch.tensor(targets_start,dtype = torch.long),
                'targets_end':torch.tensor(target_end,dtype = torch.long),                
                'sentiment':sentiment,
                'orig_tweet': self.tweet[item],                
                'orig_selected_text':self.selected_text[item],
                'offsets':torch.tensor(offsets,dtype = torch.long)
                
                }

'''
df = pd.read_csv(config.TRAINING_FILE)
ds = TweetDataset(df.text.values,df.selected_text.values)
dl = torch.utils.data.DataLoader(ds,batch_size=1,shuffle = True)

d = next(iter(dl))
d


'''




>>>>>>> 234f14c... Added app.py + final model
