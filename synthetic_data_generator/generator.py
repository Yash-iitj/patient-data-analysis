import pandas as pd
import numpy as np
from tqdm import tqdm

def generate():
    source_path = './data/final_data_small.csv'
    output_path = './data/final_data_big.csv'
    source_df = pd.read_csv(source_path)

    columns = ['CONDITION', 'ORGANIZATION', 'PROVIDER', 'PAYER', 'ENCOUNTERCLASS', 'CODE','BASE_ENCOUNTER_COST', 'TOTAL_CLAIM_COST', 'PAYER_COVERAGE','MARITAL', 'RACE', 'ETHNICITY', 'GENDER','HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME', 'IS_DEAD', 'AGE']

    source_df = source_df[columns]
    target_records = 1_000_000
    generated_rows = []
    batch_size = 100_000
    repeat_factor = target_records // batch_size

    for _ in tqdm(range(repeat_factor), desc="Generating synthetic dataset"):
        sampled = source_df.sample(n=batch_size, replace=True).copy()
        for col in ['BASE_ENCOUNTER_COST', 'TOTAL_CLAIM_COST', 'PAYER_COVERAGE','HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME']:
            sampled[col] *= np.random.uniform(0.85, 1.15, size=len(sampled))
        sampled['AGE'] = sampled['AGE'].apply(lambda x: min(100, max(0, int(x + np.random.randint(-3, 4)))))
        for col in ['CONDITION', 'CODE', 'MARITAL', 'RACE', 'ETHNICITY', 'GENDER','ORGANIZATION', 'PROVIDER', 'PAYER', 'ENCOUNTERCLASS']:
            sampled[col] = np.random.choice(source_df[col].dropna().unique(), size=len(sampled), replace=True)
        generated_rows.append(sampled)

    final_df = pd.concat(generated_rows, ignore_index=True).head(target_records)
    final_df.to_csv(output_path, index=False)