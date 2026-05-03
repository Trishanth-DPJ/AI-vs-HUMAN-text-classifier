import re
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

def extract_features(text: str) -> list:
    """
    Extracts additional handcrafted features for a single text.
    Returns a list of exactly 7 features.
    """
    slang_words = {'bro', 'fam', 'tbh', 'ngl', 'rn', 'lol', 'lmao', 'lowkey', 'cap', 'deadass', 'af', 'fr', 'yo', 'machan', 'da', 'pa', 'thambi', 'enna', 'nu', 'vibes'}
    
    # 1. Sentence length variance
    sentences = sent_tokenize(text)
    if len(sentences) == 0:
        sent_len_var = 0.0
    else:
        word_counts = [len(word_tokenize(s)) for s in sentences]
        sent_len_var = float(np.var(word_counts)) if len(word_counts) > 1 else 0.0
        
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
    inconsistency = 0.0
    if "!!" in text or "??" in text or "..." in text:
        inconsistency += 1.0
    if re.search(r'\b i \b', text): # lonely lowercase i
        inconsistency += 1.0
    if len(text) > 0 and text[-1] not in ".!?":
        inconsistency += 1.0
        
    # 6. Numeric ratio
    if char_count == 0:
        numeric_ratio = 0.0
    else:
        numeric_ratio = sum(1 for c in text if c.isdigit()) / char_count
        
    return [sent_len_var, punct_freq, cap_ratio, rep_ratio, inconsistency, numeric_ratio, slang_ratio]
