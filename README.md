# AI vs Human Text Classifier Using NLP

## Project Description
This project is an end-to-end Natural Language Processing (NLP) pipeline designed to classify input text as either **Human-written** or **AI-generated**. It utilizes traditional machine learning techniques and handcrafted textual features to provide a baseline, explainable classifier. 

It was built specifically for a college-level lab evaluation, emphasizing simple, clean, and modular code without relying on unexplainable deep-learning models.

## Objective
To build an efficient text classification model capable of distinguishing between AI-generated formal text and human-written casual text by analyzing patterns in vocabulary, sentence length, word repetition, and TF-IDF statistics.

## Technologies Used
- **Python 3.x**
- **NLTK** (Natural Language Toolkit) for text tokenization.
- **Scikit-learn** for TF-IDF Vectorization and Machine Learning Models (Logistic Regression & Naive Bayes).
- **Pandas** & **NumPy** for data manipulation.
- **Joblib** for saving and loading the trained models.
- **Streamlit** (optional) for the web UI.

## How to Run the Project

### 1. Install Dependencies
Make sure you have python installed. Run the following command to install required modules:
```bash
pip install pandas numpy scikit-learn nltk streamlit joblib scipy
```

### 2. Generate the Dataset
Since the project relies on simulated data, first run the dataset generation script:
```bash
python generate_dataset.py
```
This will create a `dataset.csv` file with 1,000 samples (500 Human / 500 AI).

### 3. Train the Model
Next, train the Logistic Regression and Naive Bayes models. This script will preprocess the text, extract features, print evaluation metrics, and export `.pkl` files.
```bash
python train.py
```

### 4. Run the Prediction Interface

**Option A: Terminal Demo (Required)**
```bash
python main.py
```
Type any text when prompted to see the prediction and confidence score.

**Option B: Web UI (Bonus)**
```bash
python -m streamlit run app.py
```
This will launch a web browser with an interactive Streamlit UI.

## Sample Input/Output

### Input 
> "This essay explores the importance of discipline and its various implications on modern society."
### Output
- **Prediction**: AI-generated
- **Confidence**: 0.94

### Input 
> "bro i am so tired right now cant even focus on this project"
### Output
- **Prediction**: Human-written
- **Confidence**: 0.78
