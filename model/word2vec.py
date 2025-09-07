from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stop words
    words = text.split()
    words = [word for word in words if word not in stop_words]
    # Rejoin
    return ' '.join(words)

p_n_dict = {}
word_embedding_dict = {}
w2v_dict = {}
file_name = '/Users/acw707/Documents/abrsm_lmth25/abrsm_lmth25.csv'
df = pd.read_csv(file_name)
# Extract only the 'performance_id' and 'mark' columns
selected = df[['performance_id', 'feedback', 'title_piece_2']]

for index, row in selected.iterrows():
    second_piece_name = selected.iloc[index]['title_piece_2']
    cleaned_peice_name = clean_text(second_piece_name)

    test_feedback = selected.iloc[index]['feedback']
    cleaned_feedback = clean_text(test_feedback)

    cleaned_feedback_per_song = cleaned_feedback.split(cleaned_peice_name)

    song_num = 0
    for fb_song in cleaned_feedback_per_song:

        song_name = selected.iloc[index]['performance_id']
        song_name = song_name + f'_{song_num + 1}'
        print("Processing song:", song_name)
        word = cleaned_feedback_per_song[song_num]
        inputs = tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        token_embeddings = outputs.last_hidden_state.squeeze(0)  # seq_len x hidden_size
        word_embedding = token_embeddings.mean(dim=0)  # combine subword tokens

        word_embedding_dict[song_name] = word_embedding.numpy()
        w2v_dict[word] = word_embedding.numpy()
        song_num += 1

np.savez('/Users/acw707/Documents/abrsm_lmth25/data/word_dict.npz', **word_embedding_dict)
np.savez('/Users/acw707/Documents/abrsm_lmth25/data/w2v_dict.npz', **w2v_dict)