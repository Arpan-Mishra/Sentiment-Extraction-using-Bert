<<<<<<< HEAD
import CONFIG as config
import torch

class BERTDataset:
    def __init__(self,tweet,sentiment):
        self.tweet = tweet
        self.sentiment = sentiment
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self,item):
        tweet =  str(self.tweet[item])
        tweet = ' '.join(tweet.split())
        sentiment = self.sentiment[item]
        
        inputs = self.tokenizer.encode_plus(
                tweet,
                None, # encode_plus can encode 2 seqs at a time we have one
                add_special_tokens = True,
                max_length = self.max_len,
                pad_to_max_length = True
                )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids'] # we will have only ones here cuz 1 sentence only
        
       
        
        return {
                'ids': torch.tensor(ids,dtype = torch.long),
                'mask': torch.tensor(mask,dtype = torch.long),
                'token_type_ids': torch.tensor(token_type_ids,dtype = torch.long),    
                'targets':torch.tensor(sentiment,dtype = torch.float),
                'orig_tweet':tweet,
                'orig_sentiment': sentiment
                }
        








=======
import CONFIG as config
import torch

class BERTDataset:
    def __init__(self,tweet,sentiment):
        self.tweet = tweet
        self.sentiment = sentiment
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self,item):
        tweet =  str(self.tweet[item])
        tweet = ' '.join(tweet.split())
        sentiment = self.sentiment[item]
        
        inputs = self.tokenizer.encode_plus(
                tweet,
                None, # encode_plus can encode 2 seqs at a time we have one
                add_special_tokens = True,
                max_length = self.max_len,
                pad_to_max_length = True
                )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids'] # we will have only ones here cuz 1 sentence only
        
       
        
        return {
                'ids': torch.tensor(ids,dtype = torch.long),
                'mask': torch.tensor(mask,dtype = torch.long),
                'token_type_ids': torch.tensor(token_type_ids,dtype = torch.long),    
                'targets':torch.tensor(sentiment,dtype = torch.float),
                'orig_tweet':tweet,
                'orig_sentiment': sentiment
                }
        








>>>>>>> 234f14c... Added app.py + final model
