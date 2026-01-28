

import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model():
    print("Starting model training...")
    
    # Load training and test data
    print("Loading datasets...")
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    X_train = train_df['content']
    y_train = train_df['label']
    X_test = test_df['content']
    y_test = test_df['label']
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize TF-IDF Vectorizer
    print("\nInitializing TF-IDF Vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=True
    )
    
    # Fit and transform training data
    print("Vectorizing text data...")
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    print(f"TF-IDF feature dimensions: {X_train_tfidf.shape[1]}")
    
    # Train Logistic Regression model
    print("\nTraining Logistic Regression classifier...")
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='liblinear'
    )
    
    classifier.fit(X_train_tfidf, y_train)
    
    # Make predictions
    print("Evaluating model performance...")
    y_pred = classifier.predict(X_test_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real News', 'Fake News']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the trained model and vectorizer
    print("\nSaving model artifacts...")
    joblib.dump(classifier, 'models/tfidf_clf.joblib')
    joblib.dump(tfidf_vectorizer, 'models/tfidf_vect.joblib')
    
    print("\nModel training completed successfully!")
    print("Saved files:")
    print("  - models/tfidf_clf.joblib")
    print("  - models/tfidf_vect.joblib")
    print("\nYou can now run the Streamlit app!")

if __name__ == '__main__':
    train_model()
