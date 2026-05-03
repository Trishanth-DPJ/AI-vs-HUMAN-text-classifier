import pandas as pd
import random
import os

TOTAL_SAMPLES = 10000

# Components to mix and match to create diversity
SUBJECTS = [
    "artificial intelligence", "the new restaurant", "my laptop", "this sci-fi movie", 
    "quantum computing", "the customer service", "this novel", "remote work", 
    "climate change policy", "the delivery guy", "my coding project", "the final exam",
    "social media algorithms", "that new phone", "the coffee shop down the street"
]

ADJECTIVES = ["amazing", "terrible", "boring", "fascinating", "ridiculous", "overrated", "groundbreaking", "unbelievable", "subpar", "excellent"]

# ==================== HUMAN TEMPLATES ====================
# Casual 
HUMAN_CASUAL = [
    "bro {subject} is literally so {adj} lol",
    "did u see {subject}? absolutely {adj}",
    "i cant even handle {subject} rn... so {adj} 😭",
    "kinda feel like {subject} is {adj} tbh",
    "okay but {subject} is actually {adj} fr fr"
]

# Formal (Academic/Review style)
HUMAN_FORMAL = [
    "The analysis of {subject} indicates a {adj} trend.",
    "Researchers have often described {subject} as {adj} in nature.",
    "In this paper, we argue that {subject} is essentially {adj}.",
    "Statistical evidence suggests that {subject} remains {adj}.",
    "The hypothesis that {subject} is {adj} was tested extensively.",
    "A thorough examination reveals that {subject} is {adj}.",
    "It is hypothesized that {subject} could be {adj}."
]

# Emotional 
HUMAN_EMOTIONAL = [
    "i am so heartbroken about how {adj} {subject} turned out.",
    "feeling so incredibly happy because {subject} is {adj}!",
    "i cried when i found out {subject} was {adj}. so moving.",
    "it makes me so angry that {subject} is this {adj}!!",
    "i am deeply moved by {subject}, it is simply {adj}."
]

# Social Media
HUMAN_SOCIAL = [
    "just thinking about {subject}... so {adj} #vibes",
    "RETWEET if you think {subject} is {adj} 🗣",
    "why is nobody talking about how {adj} {subject} is??",
    "my hot take: {subject} is {adj}. bye.",
    "literally obsessed with how {adj} {subject} is 💖"
]

# Mixed Language (Tamil + English)
HUMAN_MIXED_TAMIL = [
    "machan {subject} is so {adj} da",
    "enna bro it is {adj} about {subject}?",
    "{subject} romba {adj} irukku pa",
    "i think {subject} is {adj} nu nenaikiren",
    "sathiyama {subject} is {adj} thambi!!",
    "kandippa {subject} is very {adj}.",
    "antha {subject} vanthu romba {adj} kedaikuthu.",
    "bro {subject} epadi intha alavuku {adj} aachu?",
    "intha {subject} paathaale {adj} feel aaguthu."
]

# ==================== AI TEMPLATES ====================
# Formal (Baseline AI)
AI_FORMAL = [
    "It is important to note that {subject} serves as an unequivocally {adj} paradigm.",
    "Furthermore, the implications of {subject} are undeniably {adj} when examined closely.",
    "In conclusion, the trajectory of {subject} remains {adj} across multiple sectors."
]

# Informal Tone (AI trying to sound casual)
AI_INFORMAL = [
    "Hey there! So, {subject} is pretty {adj}, don't you think?",
    "Just wanted to chime in and say {subject} is super {adj}!",
    "Wow, I just processed some data and {subject} is totally {adj}.",
    "Honestly, looking at the facts, {subject} is quite {adj}. Cool stuff!",
    "Got it! Here is a casual reminder that {subject} is {adj}."
]

# Slang (AI simulating modern slang)
AI_SLANG = [
    "yo, {subject} is totally {adj} fam.",
    "ngl, {subject} is straight up {adj} tbh.",
    "lowkey {subject} is kinda {adj} no cap.",
    "bruh {subject} is deadass {adj}.",
    "{subject} is {adj} af, fr fr.",
    "vibin with {subject} rn, it's so {adj}.",
    "{subject} is giving {adj} energy."
]

# Imperfect Grammar (AI simulating bad grammar)
AI_IMPERFECT = [
    "{subject} is very {adj} for me to seeing.",
    "I are thinking {subject} is too {adj}.",
    "Does {subject} be {adj}?",
    "It having {adj} nature about {subject}.",
    "Many peoples say {subject} is {adj}."
]

# Shared / Ambiguous (forces model mistakes)
SHARED_AMBIGUOUS = [
    "Honestly, {subject} is so {adj}.",
    "I believe {subject} is {adj}.",
    "{subject} is really {adj}.",
    "It is clear that {subject} is {adj}.",
    "We can see {subject} is {adj}."
]

def generate_sentence(template_list):
    subj = random.choice(SUBJECTS)
    adj = random.choice(ADJECTIVES)
    text = random.choice(template_list).format(subject=subj, adj=adj)
    return text

def generate_samples(num_samples: int):
    data = []
    
    human_categories = [HUMAN_CASUAL, HUMAN_FORMAL, HUMAN_EMOTIONAL, HUMAN_SOCIAL, HUMAN_MIXED_TAMIL, SHARED_AMBIGUOUS]
    ai_categories = [AI_FORMAL, AI_INFORMAL, AI_SLANG, AI_IMPERFECT, SHARED_AMBIGUOUS]
    
    half = num_samples // 2
    
    # Generate Human texts
    for _ in range(half):
        category = random.choice(human_categories)
        text = generate_sentence(category)
        # Add some random variations
        if random.random() > 0.8: text = text.upper()
        if random.random() > 0.8: text = text.replace(".", "!!")
        if random.random() > 0.9: text = text.replace("the", "teh")
        data.append({"text": text, "label": "human"})
        
    # Generate AI texts
    for _ in range(half):
        category = random.choice(ai_categories)
        text = generate_sentence(category)
        if random.random() > 0.9: text = "As an AI model, I note that " + text[0].lower() + text[1:]
        data.append({"text": text, "label": "ai"})
        
    return data

def main():
    print(f"Generating comprehensive dataset ({TOTAL_SAMPLES} samples)...")
    dataset = generate_samples(TOTAL_SAMPLES)
    
    random.shuffle(dataset)
    
    df = pd.DataFrame(dataset)
    df.to_csv("dataset.csv", index=False)
    
    print(f"Dataset generated successfully: 'dataset.csv' with {len(df)} records.")
    print("\nDataset distribution:")
    print(df['label'].value_counts())

if __name__ == "__main__":
    main()
