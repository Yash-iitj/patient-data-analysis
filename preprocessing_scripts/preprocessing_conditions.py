import pandas as pd

# Load the data
df = pd.read_csv('./data/base_data/conditions.csv')

# Drop unnecessary columns
df.drop(columns=['SYSTEM', 'STOP', 'DESCRIPTION'], inplace=True)

# Rename columns
df.rename(columns={'CODE': 'CONDITION', 'START': 'DATE'}, inplace=True)

# Convert 'DATE' to datetime and extract only the date part
df['DATE'] = pd.to_datetime(df['DATE']).dt.date

# Display the transformed DataFrame
print(df.head())

df.to_csv('./data/conditions.csv', index=False)
