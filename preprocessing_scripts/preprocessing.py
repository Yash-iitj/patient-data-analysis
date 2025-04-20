import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def label_dropoffs_by_kmeans():
    """
    Reads './data/final_data_small.csv', adds an 'IS_DROPOFF' column using KMeans clustering,
    and overwrites the same file. The smaller cluster (between 25% and 30%) is labeled as 1 (drop-off).
    """
    file_path = "./data/final_data_small.csv"
    df = pd.read_csv(file_path)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # KMeans loop
    kmeans = KMeans(n_clusters=2)
    n = 1
    while True:
        clusters = kmeans.fit_predict(X_scaled)
        df['cluster'] = clusters
        cluster_sizes_percentage = df['cluster'].value_counts(normalize=True) * 100
        cluster_sizes_percentage.sort_index(inplace=True)
        print(f"Iteration: {n}\r", end='')
        n += 1
        if 25 < cluster_sizes_percentage.iloc[1] < 30:
            break
    print()

    # Assign drop-off label
    dropoff_cluster = cluster_sizes_percentage.idxmin()
    df['IS_DROPOFF'] = df['cluster'].apply(lambda x: 1 if x == dropoff_cluster else 0)
    df.drop(columns=['cluster'], inplace=True)

    # Overwrite the file
    df.to_csv(file_path, index=False)

def patients():
    df = pd.read_csv("./data/base_data/patients.csv")
    df.drop(columns=["SSN", 'DRIVERS', 'PASSPORT', 'PREFIX','FIRST', 'MIDDLE', 'LAST', 'SUFFIX', 'MAIDEN','BIRTHPLACE', 'ADDRESS', 'CITY', 'STATE','COUNTY', 'FIPS', 'ZIP', 'LAT', 'LON'], inplace=True)
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

def encounters():
    df = pd.read_csv("./data/base_data/encounters.csv")
    df.drop(columns=['STOP', 'REASONDESCRIPTION', 'DESCRIPTION','ORGANIZATION','PROVIDER','PAYER','CODE'], inplace=True)
    df.rename(columns={'START': 'DATE'}, inplace=True)
    df['DATE'] = pd.to_datetime(df['DATE']).dt.date
    encounter_class_map = {'urgentcare': 1,'emergency': 2,'ambulatory': 3,'inpatient': 4,'outpatient': 5,'wellness': 6,'snf': 7,'home': 8,'virtual': 9,'hospice':   10}
    df['ENCOUNTERCLASS'] = df['ENCOUNTERCLASS'].map(encounter_class_map).fillna(0).astype(int)
    df.to_csv('./data/encounters.csv', index=False)

def conditions():
    df = pd.read_csv('./data/base_data/conditions.csv')
    df.drop(columns=['SYSTEM', 'STOP', 'DESCRIPTION'], inplace=True)
    df.rename(columns={'CODE': 'CONDITION', 'START': 'DATE'}, inplace=True)
    df['DATE'] = pd.to_datetime(df['DATE']).dt.date
    df.to_csv('./data/conditions.csv', index=False)

def merge():
    patients_df = pd.read_csv('./data/patients.csv')
    conditions_df = pd.read_csv('./data/conditions.csv')
    encounters_df = pd.read_csv('./data/encounters.csv')
    merged = encounters_df.merge(patients_df, left_on='PATIENT', right_on='Id', how='left')
    merged.drop(columns=['Id_y'], inplace=True)
    merged.rename(columns={'Id_x': 'ENCOUNTER_ID'}, inplace=True)
    final_df = conditions_df.merge(merged, left_on=['ENCOUNTER', 'PATIENT'], right_on=['ENCOUNTER_ID', 'PATIENT'], how='left')
    final_df.rename(columns={'DATE_x': 'DATE_CONDITION', 'DATE_y': 'DATE_ENCOUNTER'}, inplace=True)
    final_df['PAYER_COVERAGE'] = pd.to_numeric(final_df['PAYER_COVERAGE'], errors='coerce').fillna(0)
    final_df['HEALTHCARE_COVERAGE'] = pd.to_numeric(final_df['HEALTHCARE_COVERAGE'], errors='coerce').fillna(0)
    final_df['INCOME'] = pd.to_numeric(final_df['INCOME'], errors='coerce').fillna(0)
    final_df.drop(columns=['REASONCODE', 'PATIENT', 'ENCOUNTER', 'ENCOUNTER_ID', 'DATE_CONDITION', 'DATE_ENCOUNTER'], inplace=True)
    final_df.to_csv('./data/final_data_small.csv', index=False)
    print(final_df.head())