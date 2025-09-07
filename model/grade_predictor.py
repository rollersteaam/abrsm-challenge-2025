import pandas as pd
import numpy as np

file_name = '/Users/acw707/Documents/abrsm_lmth25/abrsm_lmth25.csv'

df = pd.read_csv(file_name)

# Extract only the 'performance_id' and 'mark' columns
selected = df[['performance_id', 'ability_group']]
print(selected.iloc[0]['performance_id'], selected.iloc[0]['ability_group'])

grade_dict = {}
for index, row in selected.iterrows():
    performance_id = row['performance_id']
    grade = row['ability_group']
    print(grade)
    if grade == 'Grade 5 and above':
        one_hot_vec = np.zeros(1)
        one_hot_vec[0] = 1
    else: # Grades 1-4
        one_hot_vec = np.zeros(1)
        one_hot_vec[0] = 0
    grade_dict[performance_id + '_1'] = one_hot_vec
    grade_dict[performance_id + '_2'] = one_hot_vec

np.savez('/Users/acw707/Documents/abrsm_lmth25/data/grade_dict.npz', **grade_dict)