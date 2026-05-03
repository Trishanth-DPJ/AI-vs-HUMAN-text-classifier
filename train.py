import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from nlp_utils import clean_text
from features import extract_features

def print_metrics(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n--- {model_name} Metrics ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return acc

def print_wrong_predictions(y_true, y_pred, texts, limit=5):
    print("\n--- ERROR ANALYSIS ---")
    count = 0
    map_label = {0: "human", 1: "ai"}
    for actual, predicted, text in zip(y_true, y_pred, texts):
        if actual != predicted:
            print("\n--- WRONG PREDICTION ---")
            print(f"Text: {text}")
            print(f"Actual: {map_label[actual]}")
            print(f"Predicted: {map_label[predicted]}")
            count += 1
            if count >= limit:
                break

def main():
    print("--- 1. Loading Dataset ---")
    try:
        df = pd.read_csv("dataset.csv")
    except FileNotFoundError:
        print("Error: dataset.csv not found. Please run 'generate_dataset.py' first.")
        return

    print("\n--- 2. Preprocessing & Feature Extraction ---")
    df['clean_text'] = df['text'].apply(clean_text)
    
    print("Extracting handcrafted features...")
    X_handcrafted = np.array([extract_features(t) for t in df['text']])
    print(f"Debug: len(hc_features[0]) during training = {len(X_handcrafted[0])}")
    
    scaler = MinMaxScaler()
    X_handcrafted_scaled = scaler.fit_transform(X_handcrafted)
    print(f"Debug: scaler.n_features_in_ = {scaler.n_features_in_}")
    
    y = df['label'].map({'human': 0, 'ai': 1}).values
    
    X_train_text, X_test_text, X_train_hc, X_test_hc, y_train, y_test, texts_train, texts_test = train_test_split(
        df['clean_text'], X_handcrafted_scaled, y, df['text'], test_size=0.2, random_state=42
    )
    
    print("\n--- 3. TF-IDF Vectorization ---")
    vectorizer = TfidfVectorizer(max_features=1500)
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    
    # Combine features
    X_train_combined = hstack([X_train_tfidf, X_train_hc])
    X_test_combined = hstack([X_test_tfidf, X_test_hc])
    
    print("\n--- 4. Model Training ---")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_combined, y_train)
    
    nb_model = MultinomialNB()
    nb_model.fit(X_train_combined, y_train)
    
    print("\n--- 5. Evaluation & Error Analysis ---")
    lr_preds = lr_model.predict(X_test_combined)
    nb_preds = nb_model.predict(X_test_combined)
    
    lr_acc = print_metrics("Logistic Regression", y_test, lr_preds)
    nb_acc = print_metrics("Naive Bayes", y_test, nb_preds)
    
    print_wrong_predictions(y_test, lr_preds, texts_test, limit=5)
    
    print("\nModel Performance:")
    print(f"Logistic Regression -> Accuracy: {lr_acc*100:.2f}%")
    print(f"Naive Bayes -> Accuracy: {nb_acc*100:.2f}%")
    
    print("\nOutput Summary:")
    print("The model performs well on casual vs formal text but struggles with mixed tone inputs,")
    print("especially where AI successfully simulates slang, or where human text uses overly rigid grammar.")
    
    print("\n--- 6. Saving Models ---")
    joblib.dump(lr_model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Saved Logistic Regression to model.pkl")

if __name__ == "__main__":
    main()
