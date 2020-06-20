<<<<<<< HEAD
from tqdm import tqdm
import torch
import utils
import torch.nn as nn
import numpy as np


def loss_fn(outputs,targets):
    return nn.CrossEntropyLoss(reduction='mean')(outputs,targets)

def train_fn(dataloader,model,optimizer,device,scheduler):
    model.train()
    model.zero_grad()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    tk0 = tqdm(dataloader, total = len(dataloader))
    for bi,d in (enumerate(tk0)):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets = d['targets']
        
        ids = ids.to(device,dtype = torch.long)
        token_type_ids = token_type_ids.to(device,dtype = torch.long)
        mask = mask.to(device,dtype = torch.long)
        targets = targets.to(device,dtype = torch.long)
        
        optimizer.zero_grad()
        outputs = model(
                ids,
                mask,
                token_type_ids
                )
        
        loss = loss_fn(outputs,targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        softmax = torch.log_softmax(outputs,dim = 1)
        _,preds = torch.max(softmax,dim = 1)              
     #   print(preds)
        acc = (preds==targets).float().mean()

        accuracy.update(acc.item(),ids.size(0)) 
        losses.update(loss.item(),ids.size(0))
        tk0.set_postfix(loss = losses.avg,acc = accuracy.avg)
        
        
           
def eval_fn(dataloader,model,device):
    model.eval()
    accuracy = utils.AverageMeter()
    losses = utils.AverageMeter()
    running_acc = []
    with torch.no_grad():
        tk0 = tqdm(dataloader, total = len(dataloader))
        for bi,d in (enumerate(tk0)):
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
                loss = loss_fn(outputs,targets)
                
                softmax = torch.log_softmax(outputs,dim = 1)
                _,preds = torch.max(softmax,dim = 1)
                
                acc = (targets == preds).float().mean()
                running_acc.append(acc.cpu().detach().numpy())
                accuracy.update(acc.item(),ids.size(0))
                losses.update(loss.item(),ids.size(0))
                tk0.set_postfix(val_acc = accuracy.avg, val_loss = losses.avg)
                
        return np.mean(running_acc)
                
=======
from tqdm import tqdm
import torch
import utils
import torch.nn as nn
import numpy as np


def loss_fn(outputs,targets):
    return nn.CrossEntropyLoss(reduction='mean')(outputs,targets)

def train_fn(dataloader,model,optimizer,device,scheduler):
    model.train()
    model.zero_grad()
    losses = utils.AverageMeter()
    accuracy = utils.AverageMeter()
    tk0 = tqdm(dataloader, total = len(dataloader))
    for bi,d in (enumerate(tk0)):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets = d['targets']
        
        ids = ids.to(device,dtype = torch.long)
        token_type_ids = token_type_ids.to(device,dtype = torch.long)
        mask = mask.to(device,dtype = torch.long)
        targets = targets.to(device,dtype = torch.long)
        
        optimizer.zero_grad()
        outputs = model(
                ids,
                mask,
                token_type_ids
                )
        loss = loss_fn(outputs,targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        softmax = torch.log_softmax(outputs,dim = 1)
        _,preds = torch.max(softmax,dim = 1)              
     #   print(preds)
        acc = (preds==targets).float().mean()

        accuracy.update(acc.item(),ids.size(0)) 
        losses.update(loss.item(),ids.size(0))
        tk0.set_postfix(loss = losses.avg,acc = accuracy.avg)
        
        
           
def eval_fn(dataloader,model,device):
    model.eval()
    accuracy = utils.AverageMeter()
    losses = utils.AverageMeter()
    running_acc = []
    with torch.no_grad():
        tk0 = tqdm(dataloader, total = len(dataloader))
        for bi,d in (enumerate(tk0)):
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
                loss = loss_fn(outputs,targets)
                
                softmax = torch.log_softmax(outputs,dim = 1)
                _,preds = torch.max(softmax,dim = 1)
                
                acc = (targets == preds).float().mean()
                running_acc.append(acc.cpu().detach().numpy())
                accuracy.update(acc.item(),ids.size(0))
                losses.update(loss.item(),ids.size(0))
                tk0.set_postfix(val_acc = accuracy.avg, val_loss = losses.avg)
                
        return np.mean(running_acc)
                
>>>>>>> 234f14c... Added app.py + final model
    