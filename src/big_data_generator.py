"""
Big Data Generator for Hospital Records Mining.
Scales the Synthea dataset to create a large-scale dataset with realistic big data challenges.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class BigDataGenerator:
    """Generator for creating large-scale hospital records with realistic big data challenges."""
    
    def __init__(self, input_dir, output_dir, scale_factor=100, random_state=42):
        """
        Initialize the big data generator.
        
        Parameters:
        -----------
        input_dir : str
            Directory containing the original Synthea dataset
        output_dir : str
            Directory to save the scaled dataset
        scale_factor : int
            Factor by which to scale the dataset (default: 100)
        random_state : int
            Random seed for reproducibility
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.scale_factor = scale_factor
        self.random_state = random_state
        
        # Set random seed
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize dataframes
        self.patients_df = None
        self.encounters_df = None
        self.conditions_df = None
        self.medications_df = None
        self.observations_df = None
        self.careplans_df = None
        
        # Track generated IDs
        self.patient_ids = []
        self.encounter_ids = []
        self.provider_ids = []
        self.organization_ids = []
    
    def load_original_data(self):
        """Load the original Synthea dataset."""
        logger.info("Loading original Synthea dataset...")
        
        self.patients_df = pd.read_csv(os.path.join(self.input_dir, 'patients.csv'))
        self.encounters_df = pd.read_csv(os.path.join(self.input_dir, 'encounters.csv'))
        self.conditions_df = pd.read_csv(os.path.join(self.input_dir, 'conditions.csv'))
        self.medications_df = pd.read_csv(os.path.join(self.input_dir, 'medications.csv'))
        self.observations_df = pd.read_csv(os.path.join(self.input_dir, 'observations.csv'))
        self.careplans_df = pd.read_csv(os.path.join(self.input_dir, 'careplans.csv'))
        
        # Extract unique IDs
        self.provider_ids = self.encounters_df['PROVIDER'].unique().tolist()
        self.organization_ids = self.encounters_df['ORGANIZATION'].unique().tolist()
        
        logger.info(f"Loaded {len(self.patients_df)} patients, {len(self.encounters_df)} encounters")
    
    def generate_id(self, prefix=''):
        """Generate a unique ID."""
        return prefix + '-' + '-'.join([format(random.randint(0, 65535), 'x') for _ in range(5)])
    
    def introduce_sparsity(self, df, columns, sparsity_rate=0.2):
        """
        Introduce sparsity (missing values) in specified columns.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to modify
        columns : list
            List of columns to introduce sparsity
        sparsity_rate : float
            Probability of introducing a missing value (default: 0.2)
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with introduced sparsity
        """
        df_sparse = df.copy()
        
        for col in columns:
            if col in df_sparse.columns:
                # Generate random mask for introducing NaNs
                mask = np.random.random(len(df_sparse)) < sparsity_rate
                df_sparse.loc[mask, col] = np.nan
        
        return df_sparse
    
    def scale_patients(self):
        """Scale the patients dataset."""
        logger.info(f"Scaling patients dataset by factor of {self.scale_factor}...")
        
        # Create a list to store new patient records
        new_patients = []
        
        # Generate new patients based on original patients
        for _ in tqdm(range(self.scale_factor)):
            # Sample patients with replacement
            sampled_patients = self.patients_df.sample(n=len(self.patients_df), replace=True).copy()
            
            # Generate new IDs
            sampled_patients['Id'] = [self.generate_id('P') for _ in range(len(sampled_patients))]
            
            # Modify attributes to create variation
            # Age variation: +/- 5 years
            sampled_patients['BIRTHDATE'] = pd.to_datetime(sampled_patients['BIRTHDATE'])
            age_variation = np.random.randint(-5*365, 5*365, size=len(sampled_patients))
            sampled_patients['BIRTHDATE'] = sampled_patients['BIRTHDATE'] + pd.to_timedelta(age_variation, unit='D')
            sampled_patients['BIRTHDATE'] = sampled_patients['BIRTHDATE'].dt.strftime('%Y-%m-%d')
            
            # Vary other attributes
            if 'HEALTHCARE_EXPENSES' in sampled_patients.columns:
                sampled_patients['HEALTHCARE_EXPENSES'] *= np.random.uniform(0.7, 1.3, size=len(sampled_patients))
            
            if 'HEALTHCARE_COVERAGE' in sampled_patients.columns:
                sampled_patients['HEALTHCARE_COVERAGE'] *= np.random.uniform(0.7, 1.3, size=len(sampled_patients))
            
            if 'INCOME' in sampled_patients.columns:
                sampled_patients['INCOME'] *= np.random.uniform(0.8, 1.2, size=len(sampled_patients))
            
            # Store new patient IDs
            self.patient_ids.extend(sampled_patients['Id'].tolist())
            
            # Add to new patients list
            new_patients.append(sampled_patients)
        
        # Combine all new patients
        scaled_patients = pd.concat(new_patients, ignore_index=True)
        
        # Introduce sparsity in demographic and socioeconomic columns
        sparse_columns = ['RACE', 'ETHNICITY', 'INCOME', 'HEALTHCARE_COVERAGE', 'ADDRESS', 'CITY']
        scaled_patients = self.introduce_sparsity(scaled_patients, sparse_columns, sparsity_rate=0.15)
        
        # Save scaled patients dataset
        output_path = os.path.join(self.output_dir, 'patients_big.csv')
        scaled_patients.to_csv(output_path, index=False)
        logger.info(f"Saved {len(scaled_patients)} scaled patient records to {output_path}")
        
        return scaled_patients
    
    def scale_encounters(self, scaled_patients):
        """
        Scale the encounters dataset.
        
        Parameters:
        -----------
        scaled_patients : pandas.DataFrame
            Scaled patients dataset
        """
        logger.info("Scaling encounters dataset...")
        
        # Create a list to store new encounter records
        new_encounters = []
        
        # Process each patient
        for patient_chunk in tqdm(np.array_split(scaled_patients['Id'], 100)):  # Process in chunks
            for patient_id in patient_chunk:
                # Sample a random number of encounters for this patient
                num_encounters = np.random.poisson(5)  # Average 5 encounters per patient
                
                if num_encounters == 0:
                    continue
                
                # Sample encounters with replacement
                sampled_encounters = self.encounters_df.sample(n=num_encounters, replace=True).copy()
                
                # Generate new IDs
                sampled_encounters['Id'] = [self.generate_id('E') for _ in range(len(sampled_encounters))]
                
                # Set patient ID
                sampled_encounters['PATIENT'] = patient_id
                
                # Randomize providers and organizations
                sampled_encounters['PROVIDER'] = np.random.choice(self.provider_ids, size=len(sampled_encounters))
                sampled_encounters['ORGANIZATION'] = np.random.choice(self.organization_ids, size=len(sampled_encounters))
                
                # Create realistic temporal sequence
                base_date = datetime.now() - timedelta(days=random.randint(365, 1825))  # 1-5 years in the past
                dates = sorted([base_date + timedelta(days=random.randint(0, 1095)) for _ in range(num_encounters)])
                
                for i, date in enumerate(dates):
                    duration = random.randint(15, 240)  # 15 min to 4 hours
                    sampled_encounters.iloc[i, sampled_encounters.columns.get_loc('START')] = date.strftime('%Y-%m-%dT%H:%M:%SZ')
                    sampled_encounters.iloc[i, sampled_encounters.columns.get_loc('STOP')] = (date + timedelta(minutes=duration)).strftime('%Y-%m-%dT%H:%M:%SZ')
                
                # Vary costs
                if 'BASE_ENCOUNTER_COST' in sampled_encounters.columns:
                    sampled_encounters['BASE_ENCOUNTER_COST'] *= np.random.uniform(0.7, 1.3, size=len(sampled_encounters))
                
                if 'TOTAL_CLAIM_COST' in sampled_encounters.columns:
                    sampled_encounters['TOTAL_CLAIM_COST'] *= np.random.uniform(0.7, 1.3, size=len(sampled_encounters))
                
                # Store encounter IDs
                self.encounter_ids.extend(sampled_encounters['Id'].tolist())
                
                # Add to new encounters list
                new_encounters.append(sampled_encounters)
        
        # Combine all new encounters
        scaled_encounters = pd.concat(new_encounters, ignore_index=True)
        
        # Introduce sparsity
        sparse_columns = ['PAYER', 'PAYER_COVERAGE', 'REASONCODE', 'REASONDESCRIPTION']
        scaled_encounters = self.introduce_sparsity(scaled_encounters, sparse_columns, sparsity_rate=0.2)
        
        # Save scaled encounters dataset
        output_path = os.path.join(self.output_dir, 'encounters_big.csv')
        scaled_encounters.to_csv(output_path, index=False)
        logger.info(f"Saved {len(scaled_encounters)} scaled encounter records to {output_path}")
        
        return scaled_encounters
    
    def scale_conditions(self, scaled_patients, scaled_encounters):
        """
        Scale the conditions dataset.
        
        Parameters:
        -----------
        scaled_patients : pandas.DataFrame
            Scaled patients dataset
        scaled_encounters : pandas.DataFrame
            Scaled encounters dataset
        """
        logger.info("Scaling conditions dataset...")
        
        # Create a list to store new condition records
        new_conditions = []
        
        # Get unique conditions
        unique_conditions = self.conditions_df[['CODE', 'DESCRIPTION', 'SYSTEM']].drop_duplicates()
        
        # Process each encounter
        for encounter_chunk in tqdm(np.array_split(scaled_encounters[['Id', 'PATIENT', 'START']], 100)):  # Process in chunks
            for _, encounter in encounter_chunk.iterrows():
                # Determine number of conditions for this encounter
                num_conditions = np.random.poisson(2)  # Average 2 conditions per encounter
                
                if num_conditions == 0:
                    continue
                
                # Sample conditions
                sampled_conditions = unique_conditions.sample(n=num_conditions, replace=True).copy()
                
                # Add encounter and patient info
                sampled_conditions['PATIENT'] = encounter['PATIENT']
                sampled_conditions['ENCOUNTER'] = encounter['Id']
                sampled_conditions['START'] = encounter['START']
                sampled_conditions['STOP'] = ''  # Initialize STOP column
                
                # Determine if condition is ongoing or resolved
                for i in range(len(sampled_conditions)):
                    if np.random.random() < 0.3:  # 30% chance of being resolved
                        # Condition resolved within 1-90 days
                        start_date = pd.to_datetime(encounter['START'])
                        resolution_days = np.random.randint(1, 90)
                        stop_date = start_date + timedelta(days=resolution_days)
                        sampled_conditions.loc[sampled_conditions.index[i], 'STOP'] = stop_date.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                # Add to new conditions list
                new_conditions.append(sampled_conditions)
        
        # Combine all new conditions
        scaled_conditions = pd.concat(new_conditions, ignore_index=True)
        
        # Introduce sparsity
        sparse_columns = ['STOP']
        scaled_conditions = self.introduce_sparsity(scaled_conditions, sparse_columns, sparsity_rate=0.1)
        
        # Save scaled conditions dataset
        output_path = os.path.join(self.output_dir, 'conditions_big.csv')
        scaled_conditions.to_csv(output_path, index=False)
        logger.info(f"Saved {len(scaled_conditions)} scaled condition records to {output_path}")
        
        return scaled_conditions
    
    def generate_big_data(self):
        """Generate big data by scaling the original Synthea dataset."""
        # Load original data
        self.load_original_data()
        
        # Scale patients
        scaled_patients = self.scale_patients()
        
        # Scale encounters
        scaled_encounters = self.scale_encounters(scaled_patients)
        
        # Scale conditions
        scaled_conditions = self.scale_conditions(scaled_patients, scaled_encounters)
        
        # Scale other datasets (medications, observations, careplans)
        # Implementation similar to conditions scaling
        
        logger.info("Big data generation complete!")
        
        # Return summary
        return {
            'patients': len(scaled_patients),
            'encounters': len(scaled_encounters),
            'conditions': len(scaled_conditions)
        }


def main():
    """Main function to generate big data."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set parameters
    input_dir = '../data/synthea_sample_data_csv_latest'
    output_dir = '../data/big_data'
    scale_factor = 100  # Scale by 100x
    
    # Create generator
    generator = BigDataGenerator(input_dir, output_dir, scale_factor)
    
    # Generate big data
    summary = generator.generate_big_data()
    
    # Print summary
    logger.info(f"Generated big data summary:")
    logger.info(f"  - Patients: {summary['patients']}")
    logger.info(f"  - Encounters: {summary['encounters']}")
    logger.info(f"  - Conditions: {summary['conditions']}")


if __name__ == "__main__":
    main()
