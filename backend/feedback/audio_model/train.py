import torch
import numpy as np
import torch.nn as nn
from data_loader import Dataset_Loader_Class
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F 
import os
from combined_model import combined_model
import random
import time
import torchaudio
import soundfile as sf
print("Is MPS available?", torch.backends.mps.is_available())
print("Is MPS built?", torch.backends.mps.is_built())
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
##############################################################################
#Model Paramaters
drop = 0.15
model = combined_model(drop)
model.to(DEVICE)

#beat_model = torch.compile(beat_model, dynamic=True)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")
mark_dist = np.load('/Users/acw707/Documents/abrsm_lmth25/mark_dist.npy')
weights = 1.0 / (mark_dist + 1e-6)
weights[mark_dist == 0] = 0  # Set weight to zero for classes with no samples
weights = weights / weights.sum()  # Optional: normalize
weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

lossfn = torch.nn.CrossEntropyLoss()
#lossfn = nn.CrossEntropyLoss()
#torch.set_num_threads(80)
###############################################################################
# Optimization and Loss
TRAIN_BATCH_SIZE = 10
LEARNING_RATE = 1e-4
EPOCHS = 400

optimizer = optim.RAdam(model.parameters(), lr=LEARNING_RATE)

###############################################################################
# Data Loading 
data_dir = '/Users/acw707/Documents/abrsm_lmth25/data' 

print('Loading Data')
train_data = Dataset_Loader_Class(set_type = 'train', data_dir = data_dir)
train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
print('Train Data Loaded')
val_data = Dataset_Loader_Class(set_type = 'val', data_dir = data_dir)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
print('Val Data Loaded')

def one_hot_accuracy(pred, target):
    pred_labels = pred.argmax(dim=1)
    target_labels = target.argmax(dim=1)
    return (pred_labels == target_labels).float().mean().item()
###############################################################################
# Training and Validation
def train(model, train_loader, optimizer, device):
    num_batch = len(train_loader)
    model.train()
    running_loss = 0.0
    skip_count = 0
    for idx, (sample_all, label_all, word_embedding, track_id) in tqdm(enumerate(train_loader), total=num_batch):
        #Set data to device
        print(word_embedding.shape)
        track_id = track_id[0]

        if sample_all.max() == 0:
            #print(f'Skipping {track_id} due to error in data loading')
            continue
 
        sample_all = sample_all.to(device).float()
        label_all = label_all.to(device).float()
        
        label_all_indices = label_all.argmax(dim=1)

        pred_mark = model(sample_all)
        # print(sample_all.shape)
        # print(label_all.shape)
        #print(label_all_indices[0])
        #print(pred_mark[0].argmax())
        # print(pred_mark[0])
        loss = lossfn(pred_mark, label_all_indices)
        #Backward Pass
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        #Calculate Running Loss
        running_loss += (loss).item()
    #np.save(f'/Users/acw707/Documents/abrsm_lmth25/mark_dist.npy', mark_distribution.cpu().numpy())
    return running_loss / (len(train_loader) - skip_count)

def validate(beat_model, val_loader, device):
    num_batch = len(val_loader)
    beat_model.eval()
    running_loss = 0.0
    running_acc = 0.0
    pred_indices_list = []
    label_indices_list = []
    skip_count = 0
    with torch.no_grad():
      for idx, (sample_all, label_all, word_embedding, track_id) in tqdm(enumerate(val_loader), total=num_batch):
        #Set data to device
       #print(sample_all.shape)
        track_id = track_id[0]

        if sample_all.max() == 0:
            #print(f'Skipping {track_id} due to error in data loading')
            continue
 
        sample_all = sample_all.to(device).float()
        label_all = label_all.to(device).float()
        
        label_all_indices = label_all.argmax(dim=1)

        pred_mark = model(sample_all)
        # print(sample_all.shape)
        # print(label_all.shape)
        #print(label_all_indices[0])
        #print(pred_mark[0].argmax())
        # print(pred_mark[0])
        loss = lossfn(pred_mark, label_all_indices)

        running_loss += loss.item()
        pred_indices_list.append(pred_mark.argmax(dim=1).cpu().numpy())
        label_indices_list.append(label_all_indices.cpu().numpy())
        accuracy = one_hot_accuracy(pred_mark, label_all)
        running_acc += accuracy
    print(f'Validation Accuracy: {running_acc/(len(val_loader)-skip_count)}')
    print(pred_indices_list)
    #print(label_indices_list)
    return running_loss / (len(val_loader)-skip_count), running_acc / (len(val_loader)-skip_count)
###############################################################################
# Training Loop
best_val_loss = 10000000
saved_epoch = 0
best_acc = 0
for epoch in range(EPOCHS):

    train_loss = train(model, train_loader, optimizer, DEVICE)
    val_loss, acc = validate(model, val_loader, DEVICE)
    print(f'Epoch: {epoch+1}/{EPOCHS} Train Loss: {train_loss} Val Loss: {val_loss}')
    if acc > best_acc:
        best_acc = acc
        saved_epoch = epoch+1
        #torch.save(model.state_dict(), f'/Users/acw707/Documents/abrsm-challenge-2025/model/checkpoints/best_model.pt')
        print(f'Saved Best Model at Epoch {saved_epoch} with Val Loss: {val_loss} and Acc: {best_acc}')
        val_set_track_ids = val_data.track_ids
        #np.save('/Users/acw707/Documents/abrsm-challenge-2025/model/checkpoints/val_set_track_ids.npy', np.array(val_set_track_ids))
print(f'Best Model at Epoch {saved_epoch} with Acc: {best_acc}')


    #torch.save(model.state_dict(), f'/Users/acw707/Documents/abrsm-challenge-2025/model/checkpoints/model_epoch_{epoch+1}.pt')