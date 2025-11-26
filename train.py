import os
import pdb
import json
import torch 
import torch.nn as nn

from transformers import get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from config import getConfig


def train_and_save_model(
        model,
        train_ds,
        test_ds,
        data_collator,
        save_path,
        model_name,
):
    def data_postprocess(ds):
        ds = ds.remove_columns(["input", "label"])        # labels 추가
        ds.set_format("torch")
        return ds
    

    # save the results
    if not os.path.exists(save_path):
        os.makedirs(os.path.join(save_path, 'model'))
        os.makedirs(os.path.join(save_path, 'log'))


    # postprocessing the data
    train_ds = data_postprocess(train_ds)
    test_ds = data_postprocess(test_ds)


    # initializing the data loaders
    train_loader = DataLoader(  
        train_ds,
        shuffle = True,
        batch_size = 1,                                         ### 16
        collate_fn = data_collator,
    )

    test_loader = DataLoader(
        test_ds,
        shuffle = False,
        batch_size = 1,                                          ### 16
        collate_fn = data_collator,
    )

    # configs
    modelConfig = getConfig(model_name)

    epochs = modelConfig["epochs"]
    lr = modelConfig["lr"]

    num_steps = epochs * len(train_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_scheduler(
        "cosine",
        optimizer = optimizer,
        num_warmup_steps=0,
        num_training_steps=num_steps,
    )
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)


    best_ts_loss = float('inf')    
    log = []

    # train process      
    tr_loss = 0
    tr_count = 0

    for epoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"): 
            optimizer.zero_grad()
            
            input = batch.to(device)
            output = model(**input)

            loss = output.loss
            loss.backward()
            
            optimizer.step()
            scheduler.step()

            tr_loss += loss.item()
            tr_count += input['input_ids'].size(0)


        ts_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input = batch.to(device)
                output = model(**input)
                
                ts_loss += output.loss.item()
                
        avg_tr_loss = tr_loss/tr_count
        avg_ts_loss = ts_loss/len(test_loader)

        if avg_ts_loss < best_ts_loss:
            best_ts_loss = avg_ts_loss
            torch.save(model.state_dict(), os.path.join(save_path, 'model_weights.pth'))

        
        log_dict = {
            "epoch" : epoch,
            "train_loss" : avg_tr_loss,
            "test_loss" : avg_ts_loss,
        }
        log.append(log_dict)

        # log.jsonl로 나올 수 있게끔 각 epoch별로 {epoch, tr_loss, ts_loss} 정도로
        
        
    # log the result
    with open(os.path.join(save_path, 'log.jsonl'), "w", encoding='utf-8') as f:
        for line in log:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")
        
    
    