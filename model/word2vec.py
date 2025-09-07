from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import re

import csv 
import json
import ast

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

p_n_dict = {}
word_embedding_dict = {}
w2v_dict = {}
file_name = '/Users/acw707/Documents/abrsm_lmth25/chatgpt_abrsm_lmth25_cleaned.csv'

rows = []
with open(file_name, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter="|")
    for row in reader:
        rows.append(row) # Print the first 5 rows to check the content


for i in range(1, len(rows)):
    song_name = rows[i][0]
    #print(f'Processing {song_name}')
    for k in range(1, len(rows[i])):
        word_embed_list = []
        p_n_list = []
        song_feedback = ast.literal_eval(rows[i][k])
        for j in range(5):
            #print(song_feedback[j])
            try:
                song_word = song_feedback[j]
                print(song_word)
                p_n = song_feedback[j].split("/")[1]
                inputs = tokenizer(song_word, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                token_embeddings = outputs.last_hidden_state.squeeze(0)  # seq_len x hidden_size
                word_embedding = token_embeddings.mean(dim=0)  # combine subword tokens
                word_embed_list.append(word_embedding.numpy())
                #print(word_embedding.shape)
                if p_n == 'positive':
                    p_n_list.append(1)
                else:
                    p_n_list.append(0)
                w2v_dict[song_word] = word_embedding.numpy()
            except Exception as e:
                #print(f'Error processing {song_feedback[j]}: {e}')
                print(song_name + f'_{k}_{j}')
                continue
        print(song_name + f'_{k}')
        word_embedding_dict[song_name + f'_{k}'] = np.array(word_embed_list)
        p_n_dict[song_name  + f'_{k}'] = np.array(p_n_list)
np.savez('/Users/acw707/Documents/abrsm_lmth25/data/word_dict.npz', **word_embedding_dict)
np.savez('/Users/acw707/Documents/abrsm_lmth25/data/w2v_dict.npz', **w2v_dict)