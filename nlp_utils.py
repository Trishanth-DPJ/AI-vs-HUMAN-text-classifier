import re
import string
import numpy as np

# Download basic NLTK resources gracefully
import nltk
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
from nltk.tokenize import word_tokenize, sent_tokenize

# --- MISCLASSIFICATION PATTERN ANALYSIS ---
# Possible reasons for misclassification to watch out for:
# 1. Formal human text classified as AI: Humans writing academic or professional texts 
#    often use predictable, rigid structures similar to LLM outputs.
# 2. Informal AI text classified as human: AI models prompted with "act like a teenager" 
#    or using slang can mimic human informality and bypass basic heuristic models.
# 3. Mixed-language confusion (Tamil + English): Words not found in typical English 
#    corpora can confuse the TF-IDF vectorizer or lead to strange length variance, 
#    causing unpredictable predictions.

def clean_text(text: str) -> str:
    """
    Preprocesses the input text:
    1. Lowercasing
    2. Removal of punctuation
    3. Tokenization (and rejoin for TF-IDF)
    """
    text = text.lower()
    
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    
    tokens = word_tokenize(text)
    clean_str = ' '.join(tokens)
    
    return clean_str

def extract_features(texts: list) -> np.ndarray:
    """
    Extracts additional handcrafted features for a list of texts:
    1. Sentence length variance
    2. Punctuation frequency
    3. Capitalization ratio
    4. Word repetition ratio
    5. Grammar inconsistency
    6. Numeric ratio
    7. Slang ratio
    """
    features = []
    
    slang_words = {'bro', 'fam', 'tbh', 'ngl', 'rn', 'lol', 'lmao', 'lowkey', 'cap', 'deadass', 'af', 'fr', 'yo', 'machan', 'da', 'pa', 'thambi', 'enna', 'nu', 'vibes'}
    
    for text in texts:
        # 1. Sentence length variance
        sentences = sent_tokenize(text)
        if len(sentences) == 0:
            sent_len_var = 0.0
        else:
            word_counts = [len(word_tokenize(s)) for s in sentences]
            sent_len_var = np.var(word_counts) if len(word_counts) > 1 else 0.0
            
        # 2. Punctuation frequency (! and ?)
        char_count = len(text)
        if char_count == 0:
            punct_freq = 0.0
            cap_ratio = 0.0
        else:
            punct_count = text.count('!') + text.count('?')
            punct_freq = punct_count / char_count
            
            # 3. Capitalization ratio (uppercase / total chars)
            cap_count = sum(1 for char in text if char.isupper())
            cap_ratio = cap_count / char_count
            
        # 4. Word repetition ratio
        words = word_tokenize(text.lower())
        word_count = len(words)
        if word_count == 0:
            rep_ratio = 0.0
            slang_ratio = 0.0
        else:
            unique_words = len(set(words))
            rep_ratio = unique_words / word_count
            slang_ratio = sum(1 for w in words if w in slang_words) / word_count
            
        # 5. Grammar inconsistency score
        inconsistency = 0
        if "!!" in text or "??" in text or "..." in text:
            inconsistency += 1
        if re.search(r'\b i \b', text): # lonely lowercase i
            inconsistency += 1
        if len(text) > 0 and text[-1] not in ".!?":
            inconsistency += 1
            
        # 6. Numeric ratio
        if char_count == 0:
            numeric_ratio = 0.0
        else:
            numeric_ratio = sum(1 for c in text if c.isdigit()) / char_count
            
        features.append([sent_len_var, punct_freq, cap_ratio, rep_ratio, inconsistency, numeric_ratio, slang_ratio])
        
    return np.array(features)
