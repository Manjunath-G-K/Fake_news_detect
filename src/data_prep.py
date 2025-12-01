# # src/data_prep.py

# import pandas as pd
# from sklearn.model_selection import train_test_split
# import os

# print("--- Starting Data Preparation ---")

# # --- A. Load and Merge Data ---
# try:
#     fake_df = pd.read_csv('data/fake.csv')
#     true_df = pd.read_csv('data/True.csv')
# except FileNotFoundError as e:
#     print(f"Error: Missing input file. Make sure 'fake.csv' and 'true.csv' are in the 'data/' folder.")
#     exit()

# # Add a 'label' column: 1 for fake, 0 for real
# fake_df['label'] = 1
# true_df['label'] = 0

# combined_df = pd.concat([fake_df, true_df], ignore_index=True)

# # --- B. Cleaning and Finalizing ---
# # Drop unnecessary columns (the model only uses 'text')
# combined_df = combined_df.drop(['title', 'subject', 'date'], axis=1)

# # Clean up text and ensure no NaNs
# combined_df.dropna(inplace=True)

# # Shuffle the final dataset
# combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# print(f"Total Combined Samples: {len(combined_df)}")

# # --- C. Train-Test Split ---
# # Split the dataset into 80% Training and 20% Testing
# train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42, stratify=combined_df['label'])

# # --- D. Save Processed Files ---
# os.makedirs('data', exist_ok=True) 
# train_df.to_csv('data/train.csv', index=False)
# test_df.to_csv('data/test.csv', index=False)

# print(f"Successfully saved 'data/train.csv' ({len(train_df)} samples) and 'data/test.csv' ({len(test_df)} samples).")
# print("--- Data Preparation Complete ---")





import pandas as pd
import os
from sklearn.model_selection import train_test_split

def prepare_data():
    print("Starting data preparation...")
    
    # Load datasets
    fake_df = pd.read_csv('data/fake.csv')
    true_df = pd.read_csv('data/true.csv')
    
    print(f"Loaded {len(fake_df)} fake news articles")
    print(f"Loaded {len(true_df)} true news articles")
    
    # Add labels: 1 for fake, 0 for real
    fake_df['label'] = 1
    true_df['label'] = 0
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Create combined text from title and text
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    
    # Remove empty content
    df = df[df['content'].str.strip() != '']
    
    # Select only necessary columns
    df = df[['content', 'label']]
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Total articles after processing: {len(df)}")
    print(f"Fake news: {len(df[df['label'] == 1])}")
    print(f"Real news: {len(df[df['label'] == 0])}")
    
    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save processed datasets
    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)
    
    print(f"\nTrain set: {len(train_df)} articles")
    print(f"Test set: {len(test_df)} articles")
    print("\nData preparation completed successfully!")
    print("Files saved: data/train.csv and data/test.csv")

if __name__ == '__main__':
    prepare_data()