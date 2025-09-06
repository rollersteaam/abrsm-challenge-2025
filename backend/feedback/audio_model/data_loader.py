import numpy as np
from torch.utils.data import Dataset
import random 
import json
import os

class Dataset_Loader_Class(Dataset):
    def __init__(self, set_type = 'train', data_dir = '//Users/acw707/Documents/abrsm_lmth25/data', spec_size = 1500):
        self.set_type = set_type
        self.data_dir = data_dir
        self.spec_size = spec_size
        self.data = self.load_specs()
        self.labels = self.load_marks()
        self.track_ids = list(self.data.keys())
        self.train_ids, self.val_ids = self.train_val_split(val_fraction=0.1)
        if self.set_type == 'train':
            self.track_ids = self.train_ids
        else:
            self.track_ids = self.val_ids
        
        print(f"{self.set_type} set size: {len(self.track_ids)}")

    def load_specs(self):
        data = np.load(self.data_dir + '/specs_dict.npz', allow_pickle=True)
        return data
    
    def load_marks(self):
        marks = np.load(self.data_dir + '/mark_dict.npz', allow_pickle=True)
        return marks
    
    def train_val_split(self, val_fraction = 0.1):
        random.seed(42)
        random.shuffle(self.track_ids)
        val_size = int(len(self.track_ids) * val_fraction)
        val_ids = self.track_ids[:val_size]
        train_ids = self.track_ids[val_size:]
        return train_ids, val_ids

    def __len__(self):
        return len(self.track_ids)
        
    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        sample = self.data[track_id]
        label = self.labels[track_id]
        label = label.astype(np.int64)
        C, T, H = sample.shape
        sample = sample.mean(axis=0, keepdims=True)  # Convert to (T, H)
        sample = sample.transpose(0,2,1)  # Now (T, C, H)
        chunk_size = self.spec_size
        # Random chunk selection
        if T > chunk_size:
            start = np.random.randint(0, T - chunk_size + 1)
            end = start + chunk_size
            sample_chunk = sample[:, start:end, :]
        else:
            sample_chunk = sample[:, :chunk_size, :]
            if sample_chunk.shape[1] < chunk_size:
                pad_width = chunk_size - sample_chunk.shape[1]
                sample_chunk = np.pad(sample_chunk, ((0,0), (0,pad_width), (0,0)), mode='constant')
        # Only one chunk per song per epoch
        sample_chunk = np.expand_dims(sample_chunk, axis=0)
        repeated_label = np.expand_dims(label, axis=0)
        return sample_chunk, repeated_label, track_id
