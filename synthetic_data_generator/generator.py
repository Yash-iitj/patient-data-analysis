import pandas as pd
import numpy as np
from tqdm import tqdm

def generate():
    source_path = './data/final_data_small.csv'
    output_path = './data/final_data_big.csv'

    # Load and preserve correct column order
    df = pd.read_csv(source_path)
    columns = ['CONDITION', 'ENCOUNTERCLASS', 'BASE_ENCOUNTER_COST', 'TOTAL_CLAIM_COST',
               'PAYER_COVERAGE', 'MARITAL', 'RACE', 'ETHNICITY', 'GENDER',
               'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME', 'AGE', 'IS_DROPOFF']
    df = df[columns]

    # Parameters
    target_records = 1_000_000
    batch_size = 100_000
    repeat_factor = target_records // batch_size
    generated_rows = []

    for _ in tqdm(range(repeat_factor), desc="Generating synthetic dataset"):
        sampled = df.sample(n=batch_size, replace=True).copy()

        # Apply small numeric variation
        for col in ['BASE_ENCOUNTER_COST', 'TOTAL_CLAIM_COST', 'PAYER_COVERAGE',
                    'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME']:
            sampled[col] *= np.random.uniform(0.85, 1.15, size=len(sampled))

        # Apply small age jitter
        sampled['AGE'] = sampled['AGE'].apply(lambda x: min(100, max(0, int(x + np.random.randint(-2, 3)))))

        # Categorical variation
        for col in ['CONDITION', 'MARITAL', 'RACE', 'ETHNICITY', 'GENDER', 'ENCOUNTERCLASS']:
            sampled[col] = np.random.choice(df[col].dropna().unique(), size=len(sampled), replace=True)

        generated_rows.append(sampled)

    final_df = pd.concat(generated_rows, ignore_index=True).head(target_records)
    final_df.to_csv(output_path, index=False)
