


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
