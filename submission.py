import pandas as pd
import numpy as np
import os

def create_submission_file(predictions_file='final_predictions.txt', ids_file='test_ids.csv', output_file='titanic_submission.csv'):
    """Reads predictions and PassengerIds and creates the final submission CSV."""
    
    if not os.path.exists(predictions_file) or not os.path.exists(ids_file):
        print(f"Error: Required files ({predictions_file} and/or {ids_file}) not found.")
        print("Please run 'python3 titanic_pipeline.py' first.")
        return

    predictions = np.loadtxt(predictions_file, dtype=int)
    ids_df = pd.read_csv(ids_file)
    test_passenger_ids = ids_df['PassengerId']
    
    submission = pd.DataFrame({
        'PassengerId': test_passenger_ids,
        'Survived': predictions
    })

    submission.to_csv(output_file, index=False)
    print(f"\n✅ Submission file '{output_file}' created successfully!")
    print(submission.head())

if __name__ == '__main__':
    create_submission_file()