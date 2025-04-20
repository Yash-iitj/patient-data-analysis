from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
from joblib import load

def interface():
    app = Flask(__name__)

    MODEL_DIR = './saved_models'

    INPUT_FIELDS = [
        'CONDITION', 'ENCOUNTERCLASS', 'BASE_ENCOUNTER_COST', 'TOTAL_CLAIM_COST', 'PAYER_COVERAGE',
        'MARITAL', 'RACE', 'ETHNICITY', 'GENDER',
        'HEALTHCARE_EXPENSES', 'HEALTHCARE_COVERAGE', 'INCOME',
        'IS_DEAD', 'AGE'
    ]

    ENCODE_MAP = {
        'GENDER': {'Male': 0, 'Female': 1, 'Unknown': 2},
        'MARITAL': {'Widowed': 1, 'Divorced': 2, 'Single': 3, 'Married': 4, 'Unknown': 0},
        'RACE': {'White': 1, 'Black': 2, 'Asian': 3, 'Native': 4, 'Other': 5, 'Unknown': 0},
        'ETHNICITY': {'Hispanic': 1, 'Non-Hispanic': 2, 'Unknown': 0},
        'IS_DEAD': {'No': 0, 'Yes': 1},
        'ENCOUNTERCLASS': {
            'Urgent Care': 1, 'Emergency': 2, 'Ambulatory': 3, 'Inpatient': 4, 'Outpatient': 5,
            'Wellness': 6, 'SNF': 7, 'Home': 8, 'Virtual': 9, 'Hospice': 10
        }
    }

    SPINBOX_FIELDS = {
        'ENCOUNTERCLASS': 10
    }

    AVAILABLE_MODELS = {
        fname.replace('_model.joblib', ''): os.path.join(MODEL_DIR, fname)
        for fname in os.listdir(MODEL_DIR) if fname.endswith('.joblib')
    }

    @app.route('/', methods=['GET', 'POST'])
    def index():
        results = {}
        dropdown_options = {key: list(values.keys()) for key, values in ENCODE_MAP.items()}

        if request.method == 'POST':
            try:
                input_data = {}
                for field in INPUT_FIELDS:
                    val = request.form.get(field)
                    if field in ENCODE_MAP:
                        input_data[field] = ENCODE_MAP[field].get(val, 0)
                    elif field in SPINBOX_FIELDS:
                        input_data[field] = int(val)
                    else:
                        input_data[field] = float(val)

                input_df = pd.DataFrame([input_data])

                # If the patient is marked as deceased, override prediction
                if input_data.get('IS_DEAD') == 1:
                    for model_name in AVAILABLE_MODELS.keys():
                        results[model_name] = {
                            'prediction': 'Likely to Drop Off',
                            'probability': "100.00%"
                        }
                else:
                    # Drop IS_DEAD if it was not used during training
                    input_df_model = input_df.drop(columns=['IS_DEAD'])

                    for model_name, model_path in AVAILABLE_MODELS.items():
                        try:
                            model = load(model_path)
                            pred = model.predict(input_df_model)[0]
                            prob = model.predict_proba(input_df_model)[0][1]
                            results[model_name] = {
                                'prediction': 'Likely to Drop Off' if pred == 1 else 'Not Likely to Drop Off',
                                'probability': f"{prob * 100:.2f}%"
                            }
                        except Exception as e:
                            results[model_name] = {
                                'prediction': 'Error',
                                'probability': str(e)
                            }

            except Exception as e:
                results['error'] = {
                    'prediction': 'Input Error',
                    'probability': str(e)
                }

        return render_template(
            'index.html',
            fields=INPUT_FIELDS,
            dropdown_options=dropdown_options,
            spinbox_fields=SPINBOX_FIELDS,
            results=results
        )

    app.run(debug=True)
