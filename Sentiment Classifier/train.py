<<<<<<< HEAD
import CONFIG as config
import engine
import torch
import utils
import pandas as pd
from sklearn import model_selection
from model import BertBaseUncased
import dataset
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def run():
    print('Loading Files...')
    
    dfx = pd.read_csv(config.TRAINING_FILE).fillna('none')
    dfx = dfx[dfx['sentiment']!='neutral']
    dfx.sentiment = dfx.sentiment.apply(lambda x:utils.sent2num(x))
    
   # dfx = dfx.sample(100) 
    df_train,df_valid = model_selection.train_test_split(
            dfx,
            test_size = 0.1,
            random_state = 42,
            stratify = dfx.sentiment.values
            )
    
    df_train = df_train.reset_index(drop = True)
    df_valid = df_valid.reset_index(drop = True)
    
    print('Files loaded')
    
    train_dataset = dataset.BERTDataset( 
            tweet=df_train.text.values,
            sentiment = df_train.sentiment.values
            )
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            
            )
    
    
    valid_dataset = dataset.BERTDataset( 
            tweet=df_valid.text.values,
            sentiment = df_valid.sentiment.values
            )
    valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.VALID_BATCH_SIZE,
            )
    
    
    device = torch.device('cuda')
    print('Running on ',device)
    model = BertBaseUncased().to(device)
    
    
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.bias','layerNorm.weight']
    
    optimizer_params = [
            {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.01},
            {'params':[p for n,p in param_optimizer if  any(nd in n for nd in no_decay)],'weight_decay':0.00}
            ]
        
    
    num_training_steps = int(len(df_train)/config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_params, lr = 2e-5)
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = 0,
            num_training_steps = num_training_steps)
    
    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn( train_dataloader,model,optimizer,device,scheduler)
        accuracy = engine.eval_fn(valid_dataloader,model,device)
       
        
        if accuracy > best_accuracy:
            torch.save(model.state_dict(),config.MODEL_PATH)
            best_accuracy = accuracy
        
        torch.cuda.empty_cache()
run()    

=======
import CONFIG as config
import engine
import torch
import utils
import pandas as pd
from sklearn import model_selection
from model import BertBaseUncased
import dataset
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def run():
    print('Loading Files...')
    
    dfx = pd.read_csv(config.TRAINING_FILE).fillna('none')
    dfx = dfx[dfx['sentiment']!='neutral']
    dfx.sentiment = dfx.sentiment.apply(lambda x:utils.sent2num(x))
    
    #dfx = dfx.sample(100) 
    df_train,df_valid = model_selection.train_test_split(
            dfx,
            test_size = 0.1,
            random_state = 42,
            stratify = dfx.sentiment.values
            )
    
    df_train = df_train.reset_index(drop = True)
    df_valid = df_valid.reset_index(drop = True)
    
    print('Files loaded')
    
    train_dataset = dataset.BERTDataset( 
            tweet=df_train.text.values,
            sentiment = df_train.sentiment.values
            )
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            
            )
    
    
    valid_dataset = dataset.BERTDataset( 
            tweet=df_valid.text.values,
            sentiment = df_valid.sentiment.values
            )
    valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.VALID_BATCH_SIZE,
            )
    
    
    device = torch.device('cuda')
    print('Running on ',device)
    model = BertBaseUncased().to(device)
    
    
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.bias','layerNorm.weight']
    
    optimizer_params = [
            {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.01},
            {'params':[p for n,p in param_optimizer if  any(nd in n for nd in no_decay)],'weight_decay':0.00}
            ]
        
    
    num_training_steps = int(len(df_train)/config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_params, lr = 2e-5)
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = 0,
            num_training_steps = num_training_steps)
    
    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn( train_dataloader,model,optimizer,device,scheduler)
        accuracy = engine.eval_fn(valid_dataloader,model,device)
       
        
        if accuracy > best_accuracy:
            torch.save(model.state_dict(),config.MODEL_PATH)
            best_accuracy = accuracy
        
        torch.cuda.empty_cache()
run()    

>>>>>>> 234f14c... Added app.py + final model
 