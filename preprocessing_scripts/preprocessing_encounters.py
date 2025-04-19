import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the data
df = pd.read_csv("./data/base_data/encounters.csv")

# Drop 'STOP' column and rename 'START' to 'DATE'
df.drop(columns=['STOP', 'REASONDESCRIPTION', 'DESCRIPTION'], inplace=True)
df.rename(columns={'START': 'DATE'}, inplace=True)

# Convert 'DATE' to datetime and extract only date part
df['DATE'] = pd.to_datetime(df['DATE']).dt.date

# Label encode with NaN as 0
def label_encode_with_nan(df, column, max_labels):
    encoder = LabelEncoder()
    notna_mask = df[column].notna()
    encoded = pd.Series(np.zeros(len(df), dtype=np.int64))
    encoded[notna_mask] = encoder.fit_transform(df.loc[notna_mask, column]) + 1
    if encoded.max() > max_labels:
        print(f"Warning: {column} has more than {max_labels} unique labels.")
    return encoded

df['ORGANIZATION'] = label_encode_with_nan(df, 'ORGANIZATION', 251)
df['PROVIDER'] = label_encode_with_nan(df, 'PROVIDER', 251)
df['PAYER'] = label_encode_with_nan(df, 'PAYER', 10)
df['CODE'] = label_encode_with_nan(df, 'CODE', 46)

encounter_class_map = {'urgentcare': 1,'emergency': 2,'ambulatory': 3,'inpatient': 4,'outpatient': 5,'wellness': 6,'snf': 7,'home': 8,'virtual': 9,'hospice': 10}
df['ENCOUNTERCLASS'] = df['ENCOUNTERCLASS'].map(encounter_class_map).fillna(0).astype(int)

# Display results
print(df.head())

df.to_csv('./data/encounters.csv', index=False)