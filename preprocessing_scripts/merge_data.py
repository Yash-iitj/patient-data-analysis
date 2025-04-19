import pandas as pd

# Load data
patients_df = pd.read_csv('./data/base_data/patients.csv')
conditions_df = pd.read_csv('./data/base_data/conditions.csv')
encounters_df = pd.read_csv('./data/base_data/encounters.csv')

# Step 1: Merge encounters with patients
merged = encounters_df.merge(patients_df, left_on='PATIENT', right_on='Id', how='left')
merged.drop(columns=['Id_y'], inplace=True)  # drop duplicate patient Id
merged.rename(columns={'Id_x': 'ENCOUNTER_ID'}, inplace=True)  # rename for clarity

# Step 2: Merge the above with conditions
final_df = conditions_df.merge(merged, left_on=['ENCOUNTER', 'PATIENT'], right_on=['ENCOUNTER_ID', 'PATIENT'], how='left')

# Optional: Rename DATE columns to distinguish clearly
final_df.rename(columns={'DATE_x': 'DATE_CONDITION', 'DATE_y': 'DATE_ENCOUNTER'}, inplace=True)
# Ensure numeric before filling NaN
final_df['PAYER_COVERAGE'] = pd.to_numeric(final_df['PAYER_COVERAGE'], errors='coerce').fillna(0)
final_df['HEALTHCARE_COVERAGE'] = pd.to_numeric(final_df['HEALTHCARE_COVERAGE'], errors='coerce').fillna(0)
final_df['INCOME'] = pd.to_numeric(final_df['INCOME'], errors='coerce').fillna(0)
final_df.drop(columns=['REASONCODE', 'PATIENT', 'ENCOUNTER', 'ENCOUNTER_ID', 'DATE_CONDITION', 'DATE_ENCOUNTER'], inplace=True)

# Save for future use
final_df.to_csv('./data/final_data_small.csv', index=False)

# Preview
print(final_df.head())
