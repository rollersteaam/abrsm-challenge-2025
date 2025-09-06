import pandas as pd
import numpy as np

file_name = '/Users/acw707/Documents/abrsm_lmth25/abrsm_lmth25.csv'

df = pd.read_csv(file_name)

# Extract only the 'performance_id' and 'mark' columns
selected = df[['performance_id', 'mark']]
print(selected.iloc[0]['performance_id'], selected.iloc[0]['mark'])

mark_dict = {}
min_mark = 1000
for index, row in selected.iterrows():
    performance_id = row['performance_id']
    mark = row['mark']
    one_hot_vec = np.zeros(40)
    one_hot_vec[mark - 60] = 1
    mark_dict[performance_id + '_1'] = one_hot_vec
    mark_dict[performance_id + '_2'] = one_hot_vec

np.savez('/Users/acw707/Documents/abrsm_lmth25/data/mark_dict.npz', **mark_dict)