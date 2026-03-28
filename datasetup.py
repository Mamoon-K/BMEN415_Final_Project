import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import os
# Load training setA
def load_data():
    dfs_a = []
    for file in os.listdir('data/training/training_setA'):
        if file.endswith('.psv'):
            df = pd.read_csv(f'data/training/training_setA/{file}', sep='|')
            df['patient_id'] = file.replace('.psv', '')
            dfs_a.append(df)

    # Load setB
    dfs_b = []
    for file in os.listdir('data/training/training_setB'):
        if file.endswith('.psv'):
            df = pd.read_csv(f'data/training/training_setB/{file}', sep='|')
            df['patient_id'] = file.replace('.psv', '')
            dfs_b.append(df)

    # Combine both sets
    data = pd.concat(dfs_a + dfs_b, ignore_index=True)

    # Split by patient ID as before
    patient_ids = data['patient_id'].unique()
    train_ids, test_ids = train_test_split(patient_ids, test_size=0.2, random_state=42)
    train_ids, val_ids  = train_test_split(train_ids, test_size=0.1, random_state=42)

    train = data[data['patient_id'].isin(train_ids)]
    val   = data[data['patient_id'].isin(val_ids)]
    test  = data[data['patient_id'].isin(test_ids)]

    return train, val, test

    # checks if loading is correct
    #print(f'Total rows: {len(data)}')
    #print(f'Total patients: {data["patient_id"].nunique()}')
    #print(f'Train patients: {len(train_ids)}')
    #print(f'Val patients: {len(val_ids)}')
    #print(f'Test patients: {len(test_ids)}')


    