
import config
import engine
import torch
import pandas as pd
from sklearn import model_selection
from model import BertBaseUncased
import dataset
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

def run():
    print('Loading Files...')
    
    dfx = pd.read_csv(config.TRAINING_FILE).dropna().reset_index(drop = True)
    
    #dfx = dfx.sample(100)
    df_train,df_valid = model_selection.train_test_split(
            dfx,
            test_size = 0.1,
            random_state = 42,            
            )
    df_train = df_train.reset_index(drop = True)
    df_valid = df_valid.reset_index(drop = True)
    
    print('Files loaded')
    
    train_dataset = dataset.TweetDataset( 
            tweet=df_train.text.values,
            selected_text = df_train.selected_text.values
            )
    
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN_BATCH_SIZE,
            shuffle = False
            )
    
    
    valid_dataset = dataset.TweetDataset( 
            tweet=df_valid.text.values,
            selected_text = df_valid.selected_text.values
            )
    valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=config.VALID_BATCH_SIZE,
            shuffle = False
            )
    
    
    device = torch.device('cuda')
    print('Running on ',device)
    model = BertBaseUncased().to(device)
    
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.bias','layerNorm.weight']
    
    optimizer_params = [
            {'params':[p for n,p in param_optimizer if not any(nd in n for nd in no_decay)],'weight_decay':0.003},
            {'params':[p for n,p in param_optimizer if  any(nd in n for nd in no_decay)],'weight_decay':0.00}
            ]
    
    
    num_training_steps = int(len(df_train)/config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_params, lr = 2e-5)
    scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps = 0,
            num_training_steps = num_training_steps)
    
    
    best_jaccard = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_dataloader,model,optimizer,device,scheduler)
        jaccard = engine.eval_fn(valid_dataloader,model,device)
        
        print(f'Epochs {epoch+1}...',
              f'Jaccard {jaccard}')
        
        if jaccard > best_jaccard:
            torch.save(model.state_dict(),config.MODEL_PATH)
            best_jaccard = jaccard
        
        print('Memory Used: ',torch.cuda.memory_allocated()/1000000000,'GB')         
        torch.cuda.empty_cache()
run()
