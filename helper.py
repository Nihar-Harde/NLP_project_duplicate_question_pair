import re
import os
import pickle
import numpy as np
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
import nltk
# -------------------------------
# Load resources (only once)
# -------------------------------

BASE_DIR = os.path.dirname(__file__)

cv_path = os.path.join(BASE_DIR, 'cv.pkl')
cv = pickle.load(open(cv_path, 'rb'))

try:
    STOP_WORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOP_WORDS = set(stopwords.words('english'))
# -------------------------------
# Helper: Longest Common Substring
# -------------------------------

def longest_common_substring(s1, s2):
    if not s1 or not s2:
        return 0

    if len(s1) < len(s2):
        s1, s2 = s2, s1

    prev = [0] * (len(s2) + 1)
    longest = 0

    for i in range(1, len(s1) + 1):
        curr = [0] * (len(s2) + 1)
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
                longest = max(longest, curr[j])
        prev = curr

    return longest

# -------------------------------
# Basic Features
# -------------------------------

def test_common_words(q1, q2):
    w1 = set(word.lower().strip() for word in q1.split())
    w2 = set(word.lower().strip() for word in q2.split())
    return len(w1 & w2)

def test_total_words(q1, q2):
    w1 = set(word.lower().strip() for word in q1.split())
    w2 = set(word.lower().strip() for word in q2.split())
    return len(w1) + len(w2)

# -------------------------------
# Token Features
# -------------------------------

def test_fetch_token_features(q1, q2):
    SAFE_DIV = 0.0001
    token_features = [0.0] * 8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if not q1_tokens or not q2_tokens:
        return token_features

    q1_words = set(w for w in q1_tokens if w not in STOP_WORDS)
    q2_words = set(w for w in q2_tokens if w not in STOP_WORDS)

    q1_stops = set(w for w in q1_tokens if w in STOP_WORDS)
    q2_stops = set(w for w in q2_tokens if w in STOP_WORDS)

    common_words = len(q1_words & q2_words)
    common_stops = len(q1_stops & q2_stops)
    common_tokens = len(set(q1_tokens) & set(q2_tokens))

    token_features[0] = common_words / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_words / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stops / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stops / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_tokens / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_tokens / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features

# -------------------------------
# Length Features
# -------------------------------

def test_fetch_length_features(q1, q2):
    length_features = [0.0] * 3

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if not q1_tokens or not q2_tokens:
        return length_features

    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    lcs_len = longest_common_substring(q1, q2)
    length_features[2] = lcs_len / (min(len(q1), len(q2)) + 1)

    return length_features

# -------------------------------
# Fuzzy Features
# -------------------------------

# def test_fetch_fuzzy_features(q1, q2):
#     return [
#         fuzz.QRatio(q1, q2),
#         fuzz.partial_ratio(q1, q2),
#         fuzz.token_sort_ratio(q1, q2),
#         fuzz.token_set_ratio(q1, q2)
#     ]

# -------------------------------
# Preprocessing
# -------------------------------

def preprocess(q):
    q = str(q).lower().strip()

    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')
    q = q.replace('[math]', '')

    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Remove HTML
    q = BeautifulSoup(q, "html.parser").get_text()

    # Remove punctuation
    q = re.sub(r'\W', ' ', q).strip()

    return q

# -------------------------------
# Final Query Builder
# -------------------------------

def query_point_creator(q1, q2):
    input_query = []


    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # Basic features
    input_query.extend([
        len(q1),
        len(q2),
        len(q1.split()),
        len(q2.split()),
        test_common_words(q1, q2),
        test_total_words(q1, q2),
        round(test_common_words(q1, q2) / (test_total_words(q1, q2) + 0.0001), 2)
    ])

    # Token features
    input_query.extend(test_fetch_token_features(q1, q2))

    # Length features
    input_query.extend(test_fetch_length_features(q1, q2))

    # # Fuzzy features
    # input_query.extend(test_fetch_fuzzy_features(q1, q2))

    # Vectorization
    q1_bow = cv.transform([q1]).toarray()
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, 18), q1_bow, q2_bow))