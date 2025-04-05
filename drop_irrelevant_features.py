import pandas as pd
import argparse
import os
from datetime import datetime

def drop_irrelevant_features(input_file, output_file):
    # Load the dataset
    print(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # List of irrelevant features to drop
    irrelevant_features = [
        'file_name',        # Just an identifier
        'package_name',     # Too specific, could cause overfitting
        'app_name',         # Too specific, could cause overfitting
        'processing_time',  # Metadata about extraction process, not the app
        'error',            # Information about extraction process
        'jni_calls',        # Text data that needs special processing
        'version_name'      # Developer-assigned value, not predictive
    ]
    
    # Only drop features that exist in the dataset
    features_to_drop = [col for col in irrelevant_features if col in df.columns]
    
    # Drop the features
    print(f"Dropping {len(features_to_drop)} irrelevant features: {', '.join(features_to_drop)}")
    df_cleaned = df.drop(columns=features_to_drop)
    
    # Save the cleaned dataset
    print(f"Saving cleaned dataset to {output_file}...")
    df_cleaned.to_csv(output_file, index=False)
    
    print(f"Cleaned dataset saved with {df_cleaned.shape[1]} features")
    print(f"Removed features: {len(features_to_drop)}")
    print(f"Remaining features: {df_cleaned.shape[1]}")
    
    return df_cleaned

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Remove irrelevant features from Android malware dataset')
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', help='Output CSV file')
    
    args = parser.parse_args()
    
    # If output file not specified, create one with timestamp
    if not args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"{filename}_cleaned_{timestamp}.csv"
    
    # Drop irrelevant features
    drop_irrelevant_features(args.input, args.output)

if __name__ == "__main__":
    main()
