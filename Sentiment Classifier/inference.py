<<<<<<< HEAD
import torch
import utils
import dataset
import pandas as pd
from model import BertBaseUncased
import CONFIG as config
from tqdm import tqdm


def test_fn(dataloader,model,device):
    model.eval()
    accuracy = utils.AverageMeter()
    fin_outputs = []
    tk0 = tqdm(dataloader,total = len(dataloader))
    with torch.no_grad():         
        for bi,d in enumerate(tk0):
            ids = d['ids']
            token_type_ids = d['token_type_ids']
            mask = d['mask']
            targets = d['targets']
            
            
            ids = ids.to(device,dtype = torch.long)
            token_type_ids = token_type_ids.to(device,dtype = torch.long)
            mask = mask.to(device,dtype = torch.long)
            targets = targets.to(device,dtype = torch.long)
            
           
            outputs = model(
                       ids,
                       mask,
                       token_type_ids
                       )
             
            outputs = outputs.float()
            softmax = torch.log_softmax(outputs,dim = 1)
            _,preds = torch.max(softmax,dim = 1)
            
            fin_outputs.extend(preds)
                
            acc = (targets == preds).float().mean()
            accuracy.update(acc.item(),ids.size(0))
            tk0.set_postfix(test_acc = accuracy.avg)            
            
        return fin_outputs
    

def run_test():
    df = pd.read_csv(config.TESTING_FILE)
    df = df[df.sentiment!='neutral']
    df.sentiment = df.sentiment.apply(lambda x:utils.sent2num(x))

    
    test_dataset = dataset.BERTDataset( 
            tweet=df.text.values,
            sentiment = df.sentiment.values
            )
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.VALID_BATCH_SIZE,            
            )
    
    device = 'cpu'
    model = BertBaseUncased().to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    
    outputs = test_fn(test_dataloader,model,device)
    
    print('Test Accuracy: ',(outputs == df.sentiment.values).mean())
    
run_test()
=======
import torch
import utils
import dataset
import pandas as pd
from model import BertBaseUncased
import CONFIG as config
from tqdm import tqdm


def test_fn(dataloader,model,device):
    model.eval()
    accuracy = utils.AverageMeter()
    fin_outputs = []
    tk0 = tqdm(dataloader,total = len(dataloader))
    with torch.no_grad():         
        for bi,d in enumerate(tk0):
            ids = d['ids']
            token_type_ids = d['token_type_ids']
            mask = d['mask']
            targets = d['targets']
            
            
            ids = ids.to(device,dtype = torch.long)
            token_type_ids = token_type_ids.to(device,dtype = torch.long)
            mask = mask.to(device,dtype = torch.long)
            targets = targets.to(device,dtype = torch.long)
            
           
            outputs = model(
                       ids,
                       mask,
                       token_type_ids
                       )
             
            outputs = outputs.float()
            softmax = torch.log_softmax(outputs,dim = 1)
            _,preds = torch.max(softmax,dim = 1)
            
            fin_outputs.extend(preds)
                
            acc = (targets == preds).float().mean()
            accuracy.update(acc.item(),ids.size(0))
            tk0.set_postfix(test_acc = accuracy.avg)            
            
        return fin_outputs
    

def run_test():
    df = pd.read_csv(config.TESTING_FILE)
    df = df[df.sentiment!='neutral']
    df.sentiment = df.sentiment.apply(lambda x:utils.sent2num(x))

    
    test_dataset = dataset.BERTDataset( 
            tweet=df.text.values,
            sentiment = df.sentiment.values
            )
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.VALID_BATCH_SIZE,            
            )
    
    device = 'cpu'
    model = BertBaseUncased().to(device)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    
    outputs = test_fn(test_dataloader,model,device)
    
    print('Test Accuracy: ',(outputs == df.sentiment.values).mean())
    
run_test()
>>>>>>> 234f14c... Added app.py + final model
    