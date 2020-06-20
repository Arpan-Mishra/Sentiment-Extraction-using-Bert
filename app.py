import config
import torch
import utils
import numpy as np

from flask import Flask
from flask import request,render_template
from model import BertBaseUncased,Sentiment

app = Flask(__name__)

def sentence_prediction(sentence):
    tokenizer1 = config.Plrty_Tokenizer
    tokenizer2 = config.TOKENIZER
    max_len = config.MAX_LEN
    
    tweet = str(sentence)
    tweet_enc = utils.preprocess_tweet(tweet,tokenizer1,max_len,device)
    out = Polarity_Model(tweet_enc['ids'].view(1,-1),tweet_enc['mask'].view(1,-1),tweet_enc['type_ids'].view(1,-1))
         
    _,pred = torch.max(torch.softmax(out,dim=1),dim=1)
        
        
    sentiment = 'negative'
    if pred==1:
        sentiment = 'positive'
     
    print('Predicted Sentiment: ',sentiment)    
    
    sentiment_id = {
            'positive': 3893,
            'negative':4997                
            }
    
    tok_tweet = tokenizer2.encode(tweet)
    tweet_ids = tok_tweet.ids[1:-1]
    tok_tweet_offsets = tok_tweet.offsets[1:-1] 

    tok_tweet_ids = [101] + [sentiment_id[sentiment]] + [102] + tweet_ids + [102]
    tok_type_ids = [0,0,0] + [1]*len(tweet_ids) + [1]
    mask = [1]*len(tok_type_ids)
    tok_tweet_offsets = [(0,0)]*3 + tok_tweet_offsets + [(0,0)]
     
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
         
    
    ids = torch.tensor(ids,dtype = torch.long).unsqueeze(0)
    mask = torch.tensor(mask,dtype = torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(tok_type_ids,dtype = torch.long).unsqueeze(0)
    offsets = torch.tensor(offsets,dtype = torch.long)
    
    
    ids = ids.to(device,dtype = torch.long)
    token_type_ids = token_type_ids.to(device,dtype = torch.long)
    mask = mask.to(device,dtype = torch.long)
    
     
    out_start,out_end = bert_model(
                   ids,
                   mask,
                   token_type_ids
                   )
   
    
    out_start = torch.softmax(out_start,dim = 1).cpu().detach().numpy()
    out_end = torch.softmax(out_end,dim = 1).cpu().detach().numpy()

    idx_start = np.argmax(out_start)
    idx_end = np.argmax(out_end)
    selected_text = "random"
    
    print(idx_start,idx_end)
    
    final_text,_ = utils.calculate_jaccard(tweet,offsets,selected_text,idx_start,idx_end)
    
    return sentiment,final_text
    


@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        sentence = request.form['text']
        sentiment,extracted_phrase = sentence_prediction(sentence)    
        
        return render_template('result.html',sentiment = sentiment, 
                               extracted_phrase = extracted_phrase)


if __name__=='__main__':
    device = 'cpu'
    Polarity_Model = Sentiment().to(device)
    Polarity_Model.load_state_dict(torch.load(config.SENTIMENT_MODEL_PATH))
    bert_model = BertBaseUncased().to(device)
    bert_model.load_state_dict(torch.load(config.MODEL_PATH))
    Polarity_Model.eval()
    bert_model.eval()
    app.run()







