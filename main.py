import joblib
import warnings
from scipy.sparse import hstack
from nlp_utils import clean_text, get_handcrafted_features

warnings.filterwarnings("ignore")

def predict_text(text: str, model, vectorizer, scaler):
    cleaned = clean_text(text)
    
    hc_features = get_handcrafted_features([text])
    hc_scaled = scaler.transform(hc_features)
    
    tfidf_features = vectorizer.transform([cleaned])
    combined_features = hstack([tfidf_features, hc_scaled])
    
    prediction = model.predict(combined_features)[0]
    probabilities = model.predict_proba(combined_features)[0]
    
    label = "AI-generated" if prediction == 1 else "Human-written"
    confidence = probabilities[prediction]
    
    return label, confidence

def main():
    print("===========================================")
    print("   AI vs Human Text Classifier (Terminal)  ")
    print("===========================================\n")
    
    try:
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        print("Error: Required .pkl files not found. Please run 'python train.py' first.")
        return

    print("Model loaded successfully. Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_input = input("Enter text to classify:\n> ")
            if user_input.strip().lower() in ['exit', 'quit']:
                print("Exiting...")
                break
                
            if not user_input.strip():
                continue
                
            label, confidence = predict_text(user_input, model, vectorizer, scaler)
            
            # Formatted exactly as requested
            print(f"\n{label} (Confidence: {confidence:.2f})\n")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
