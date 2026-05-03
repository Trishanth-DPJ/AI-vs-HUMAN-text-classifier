import streamlit as st
import joblib
import warnings

# We import the predict_text function from our main module
from main import predict_text

warnings.filterwarnings("ignore")

def load_objects():
    try:
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, vectorizer, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please run 'python train.py' in the terminal first.")
        return None, None, None

def main():
    st.set_page_config(page_title="AI vs Human text Classifier", page_icon="🤖")
    
    st.title("🤖 AI vs Human Text Classifier")
    st.markdown("This project classifies input text as either **Human-written** or **AI-generated** using Natural Language Processing and Logistic Regression.")
    
    st.sidebar.header("About this Project")
    st.sidebar.info(
        "**Features Used:**\n"
        "1. TF-IDF (Term Frequency-Inverse Document Frequency)\n"
        "2. Average Sentence Length\n"
        "3. Word Count\n"
        "4. Repetition Ratio\n"
        "5. Punctuation Frequency\n"
        "6. Capitalization Variation"
    )
    st.sidebar.caption("Built for college NLP project.")
    
    model, vectorizer, scaler = load_objects()
    
    if model and vectorizer and scaler:
        user_text = st.text_area("Enter the text you want to classify:", height=150, placeholder="Example: This essay explores the importance of discipline...")
        
        if st.button("Classify Text"):
            if user_text.strip():
                with st.spinner("Analyzing text..."):
                    label, confidence = predict_text(user_text, model, vectorizer, scaler)
                
                st.subheader("Result:")
                if label == "AI-generated":
                    st.error(f"**{label}**")
                else:
                    st.success(f"**{label}**")
                    
                st.write(f"**Confidence Score:** {confidence:.2f}")
                
            else:
                st.warning("Please enter some text to classify.")

if __name__ == "__main__":
    main()
