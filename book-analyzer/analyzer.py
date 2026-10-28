import spacy
import nltk
from collections import Counter
from textstat import flesch_reading_ease

# Download NLTK resources if not already
nltk.download('punkt', quiet=True)

nlp = spacy.load("en_core_web_sm")

def analyze_text(text):
    results = {}

    # Theme detection (keywords)
    doc = nlp(text)
    keywords = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    common_keywords = Counter(keywords).most_common(10)
    results['keywords'] = common_keywords

    # Summary (first 3 sentences)
    sentences = nltk.sent_tokenize(text)
    results['summary'] = " ".join(sentences[:3])

    # Quotes (lines with quotation marks)
    quotes = [s for s in sentences if '"' in s or "'" in s]
    results['quotes'] = quotes[:5]

    # Readability score
    results['readability'] = flesch_reading_ease(text)

    return results
