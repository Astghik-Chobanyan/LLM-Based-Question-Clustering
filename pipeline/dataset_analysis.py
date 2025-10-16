from pathlib import Path

import pandas as pd

# Get the project root directory
project_root = Path(__file__).parent.parent
data_path = project_root / "data" / "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"

def explore_dataset():
    print(f"Reading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    print("\nDataset Info:")
    print("-" * 50)
    print(f"Total number of rows: {len(df)}")
    print(f"Columns in dataset: {', '.join(df.columns)}")
    
    print("\nUnique values by category:")
    print("-" * 50)
    for column in df.columns:
        unique_count = df[column].nunique()
        print(f"\n{column}:")
        print(f"Number of unique values: {unique_count}")
        
        if unique_count < 20:
            value_counts = df[column].value_counts()
            print("\nValue distribution:")
            for value, count in value_counts.items():
                print(f"  {value}: {count}")

if __name__ == "__main__":
    explore_dataset()
