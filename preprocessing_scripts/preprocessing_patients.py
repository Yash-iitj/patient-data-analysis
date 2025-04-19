import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv("./data/base_data/patients.csv")
df.drop(columns=["SSN", 'DRIVERS', 'PASSPORT', 'PREFIX','FIRST', 'MIDDLE', 'LAST', 'SUFFIX', 'MAIDEN','BIRTHPLACE', 'ADDRESS', 'CITY', 'STATE','COUNTY', 'FIPS', 'ZIP', 'LAT', 'LON'], inplace=True)

df['IS_DEAD'] = df['DEATHDATE'].notna().astype(int)

df.drop('DEATHDATE', inplace=True, axis=1)

df['MARITAL'] = df['MARITAL'].map({'W': 1, 'D': 2, 'S': 3, 'M': 4}).fillna(0).astype(int)
df['GENDER'] = df['GENDER'].map({'M': 0, 'F': 1}).fillna(2).astype(int)
df['ETHNICITY'] = df['ETHNICITY'].map({'hispanic': 1,'nonhispanic': 2}).fillna(0).astype(int)
df['RACE'] = df['RACE'].map({'white': 1,'black': 2,'asian': 3,'native': 4,'other': 5}).fillna(0).astype(int)

df['BIRTHDATE'] = pd.to_datetime(df['BIRTHDATE'], errors='coerce')
today = pd.Timestamp(datetime.today())
df['AGE'] = ((today - df['BIRTHDATE']).dt.days / 365.25).astype(int)
df.drop(columns=['BIRTHDATE'], inplace=True)

df.to_csv("./data/patients.csv", index=False)