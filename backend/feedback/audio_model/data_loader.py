import numpy as np
from torch.utils.data import Dataset
import random 
import json
import os
import torchaudio
import torch.nn.functional as F
import torch
class Dataset_Loader_Class(Dataset):
    def __init__(self, set_type = 'train', data_dir = '//Users/acw707/Documents/abrsm_lmth25/data'):
        self.set_type = set_type
        self.data_dir = data_dir
        self.data = self.load_specs()
        self.labels = self.load_marks()
        self.grades = self.load_grades()
        self.word_embeddings = self.load_word_embeddings()
        self.track_ids = list(self.data.keys())
        # Calculate rare_class_index from training set
        self.train_ids, self.val_ids = self.train_val_split(val_fraction=0.1)
        if self.set_type == 'train':
            self.track_ids = self.train_ids
        else:
            self.track_ids = self.val_ids
            
        # num_classes = self.labels[self.track_ids[0]].shape[0]
        # class_counts = np.zeros(num_classes, dtype=int)
        # for track_id in self.track_ids:
        #     label = self.labels[track_id]
        #     class_idx = np.argmax(label)
        #     class_counts[class_idx] += 1
        # self.rare_class_index = class_counts.tolist()
        # if set_type == 'train':
        #     oversampled_ids = []
        #     max_count = max(self.rare_class_index)
        #     for track_id in self.track_ids:
        #         label = self.labels[track_id]
        #         class_idx = np.argmax(label)
        #         count = self.rare_class_index[class_idx]
        #         repeat = max_count // count if count > 0 else 1
        #         oversampled_ids.extend([track_id] * repeat)
        #     self.track_ids = oversampled_ids
        
        print(f"{self.set_type} set size: {len(self.track_ids)}")

    def load_specs(self):
        data = np.load(self.data_dir + '/emb_dict.npz', allow_pickle=True)
        return data
    
    def load_marks(self):
        marks = np.load(self.data_dir + '/mark_dict.npz', allow_pickle=True)
        return marks
    def load_grades(self):
        grades = np.load(self.data_dir + '/grade_dict.npz', allow_pickle=True)
        return grades
    
    def load_word_embeddings(self):
        word_dict = np.load(self.data_dir + '/word_dict.npz', allow_pickle=True)
        return word_dict
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
        grade = self.grades[track_id]
        grade = grade.astype(np.float32)
        word_embedding = self.word_embeddings[track_id]
        sample = sample.mean(axis=0)  # Convert to (T, H)
        
        sample = np.concatenate((sample, grade), axis = 0)  # Concatenate along feature dimension
        #print(sample.shape, label.shape, track_id)
        return sample, label, word_embedding, track_id
