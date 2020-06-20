<<<<<<< HEAD
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def calculate_jaccard(tweet,offset,selected_text,idx_start,idx_end,sentiment):
    
    final_output = ""
    
    if idx_start>idx_end:
        idx_start = idx_end

    for idx in range(idx_start,idx_end+1):
        final_output+=tweet[offset[idx][0]:offset[idx][1]]
        if (idx+1)<len(offset) and offset[idx][1] < offset[idx+1][0]:
            final_output+= " "
        
     
    if  sentiment=='neutral' or len(tweet.split())<2:
         final_output = tweet
                         
    return final_output,jaccard(selected_text.strip(),final_output.strip())
=======
import torch
class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def calculate_jaccard(tweet,offset,selected_text,idx_start,idx_end):
    
    final_output = ""
    
    if idx_start>idx_end:
        idx_start = idx_end

    for idx in range(idx_start,idx_end+1):
        final_output+=tweet[offset[idx][0]:offset[idx][1]]
        if (idx+1)<len(offset) and offset[idx][1] < offset[idx+1][0]:
            final_output+= " "
         
    return final_output,jaccard(selected_text.strip(),final_output.strip())

def preprocess_tweet(tweet,tokenizer,max_len,device):
    tweet_tok = tokenizer.encode_plus(tweet,
                                      None,
                                      add_special_tokens = True,
                                      max_length = max_len,
                                      pad_to_max_length = True)
    
    ids = tweet_tok.input_ids
    mask = tweet_tok.attention_mask
    type_ids = tweet_tok.token_type_ids
    
    return {
            'ids':torch.tensor(ids,dtype = torch.long).to(device),
            'mask':torch.tensor(mask,dtype = torch.long).to(device),
            'type_ids': torch.tensor(type_ids,dtype = torch.long).to(device)
            }





>>>>>>> 234f14c... Added app.py + final model
