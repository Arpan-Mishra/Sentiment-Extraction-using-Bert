<<<<<<< HEAD
import CONFIG as config
import transformers
import torch.nn as nn
import torch
import torch.nn.functional as F


class BertBaseUncased(nn.Module):
    def __init__(self):
        super(BertBaseUncased,self).__init__()
        c = transformers.BertConfig.from_pretrained(config.BERT_PATH,output_hidden_states = True)        
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH,config = c)
       
                
        self.bert_drop = nn.Dropout(0.4)
        self.linear = nn.Linear(768,3)
        self.linear_drop = nn.Dropout(0.3)
        
       # self.out = nn.Linear(100,3)  
        
    def forward(self, ids,mask,token_type_ids):
        o1,o2,o3 = self.bert(ids,
                              mask,
                              token_type_ids)
        
        
        
        o = torch.stack(o3[-4:]).sum(0) # taking sum of the first hidden state of last 4 layers 
        o = torch.mean(o,1)
        o = self.bert_drop(o)
                       
        o = self.linear(o)
        o = self.linear_drop(o)
        
         
       # output = self.out(o)
        
=======
import CONFIG as config
import transformers
import torch.nn as nn
import torch


class BertBaseUncased(nn.Module):
    def __init__(self):
        super(BertBaseUncased,self).__init__()
        c = transformers.BertConfig.from_pretrained(config.BERT_PATH,output_hidden_states = True)        
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH,config = c)
       
                
        self.bert_drop = nn.Dropout(0.4)
        self.linear = nn.Linear(768,2)
        self.linear_drop = nn.Dropout(0.3)
        
       # self.out = nn.Linear(100,2)  
        
    def forward(self, ids,mask,token_type_ids):
        o1,o2,o3 = self.bert(ids,
                              mask,
                              token_type_ids)
        
        
        
        o = torch.stack(o3[-4:]).sum(0) # taking sum of the first hidden state of last 4 layers 
        o = torch.mean(o,1)
        o = self.bert_drop(o)
                       
        o = self.linear(o)
        o = self.linear_drop(o)
        
         
       # output = self.out(o)
        
>>>>>>> 234f14c... Added app.py + final model
        return o