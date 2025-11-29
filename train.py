import os
import gc
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
        mode_flag,
):
    def data_postprocess(ds):
        ds = ds.remove_columns(["input", "label", "answer"])        # labels 추가
        ds.set_format("torch")
        return ds

    # create the save_path
    os.makedirs(os.path.join(save_path, 'model'), exist_ok = True)
    os.makedirs(os.path.join(save_path, 'log'), exist_ok = True)

    # set the model info
    model_li = model_name.split('/')
    model_info = '-'.join(model_li)


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
    #loss_fn = nn.CrossEntropyLoss()
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
    step = 0

    for epoch in range(epochs):

        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"): 
            optimizer.zero_grad()
            
            input = {k: v.to(device) for k, v in batch.items()}
            output = model(**input)

            loss = output.loss
            loss.backward()
            
            optimizer.step()
            scheduler.step()

            tr_loss += loss.item()
            tr_count += input['input_ids'].size(0)
            step += 1

            if step % 10 == 0:           ## logging the train_loss at every 10 steps
                step_log = {
                    "step" : step,
                    "train_loss" : tr_loss/tr_count
                }
                log.append(step_log)



        ts_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                input = {k: v.to(device) for k, v in batch.items()}
                output = model(**input)
                
                ts_loss += output.loss.item()
                

        avg_tr_loss = tr_loss/tr_count
        avg_ts_loss = ts_loss/len(test_loader)

        if avg_ts_loss < best_ts_loss:
            best_ts_loss = avg_ts_loss
            best_model_pth = os.path.join(save_path, 'model', f'model_weights_{model_info}')
            model.save_pretrained(best_model_pth)

        epoch_log = {
            "epoch" : epoch+1,
            "train_loss" : avg_tr_loss,
            "test_loss" : avg_ts_loss,
        }
        log.append(epoch_log)

        # log.jsonl로 나올 수 있게끔 각 epoch별로 {epoch, tr_loss, ts_loss} 정도로
        
        gc.collect()
        torch.cuda.empty_cache()


    # log the result
    with open(os.path.join(save_path, 'log', f'log_{model_info}_epochs:{epochs}_lr:{lr}_({mode_flag}).jsonl'), "w", encoding='utf-8') as f:
        for line in log:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")


    return best_model_pth      
    
    