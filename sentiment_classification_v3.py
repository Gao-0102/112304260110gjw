import pandas as pd
import numpy as np
import re
import pickle
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = text.lower()
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"[^a-zA-Z\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = text.split()
    
    stops = set(stopwords.words("english"))
    negation_words = {'not', 'no', 'never', 'nor', 'none', 'hardly', 'scarcely', 'barely', 'cannot', 'isn', 'wasn', 'weren', 'haven', 'hasn', 'hadn', 'don', 'doesn', 'didn', 'shouldn', 'wouldn', 'couldn', 'mightn', 'mustn'}
    filtered_tokens = [w for w in tokens if w not in stops or w in negation_words]
    
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]
    
    return " ".join(lemmatized_tokens)

def add_custom_features(df):
    df['review_len'] = df['cleaned_review'].apply(lambda x: len(x.split()))
    df['sentiment_word_count'] = df['cleaned_review'].apply(count_sentiment_words)
    df['exclamation_count'] = df['review'].apply(lambda x: x.count('!'))
    df['question_count'] = df['review'].apply(lambda x: x.count('?'))
    df['caps_ratio'] = df['review'].apply(calculate_caps_ratio)
    return df

positive_words = set(['great', 'excellent', 'perfect', 'wonderful', 'best', 'amazing', 'love', 'loved', 'like', 'good', 'nice', 'fun', 'enjoy', 'enjoyed', 'happy', 'beautiful', 'awesome', 'fantastic', 'brilliant', 'superb', 'outstanding', 'remarkable', 'terrific', 'fabulous', 'delightful', 'impressive', 'incredible', 'marvelous', 'splendid'])
negative_words = set(['bad', 'worst', 'awful', 'terrible', 'horrible', 'disappointing', 'disappointed', 'boring', 'dull', 'poor', 'sad', 'hate', 'hated', 'waste', 'pathetic', 'ridiculous', 'stupid', 'annoying', 'frustrating', 'awful', 'atrocious', 'dreadful', 'abysmal', 'pathetic', 'embarrassing'])

def count_sentiment_words(text):
    words = text.split()
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    return pos_count - neg_count

def calculate_caps_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return 0.0
    caps = [c for c in letters if c.isupper()]
    return len(caps) / len(letters)

def main():
    print("="*60)
    print("Step 1: Loading and preprocessing data")
    print("="*60)

    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

    try:
        WordNetLemmatizer()
    except LookupError:
        nltk.download('wordnet')

    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

    print(f"Train size: {len(train)}, Test size: {len(test)}")

    print("\nPreprocessing training reviews...")
    train['cleaned_review'] = train['review'].apply(preprocess_text)

    print("Preprocessing test reviews...")
    test['cleaned_review'] = test['review'].apply(preprocess_text)

    print("\nAdding custom features...")
    train = add_custom_features(train)
    test = add_custom_features(test)

    print("\n" + "="*60)
    print("Step 2: Creating features")
    print("="*60)

    tfidf1 = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        analyzer='word'
    )

    tfidf2 = TfidfVectorizer(
        ngram_range=(2, 3),
        max_features=5000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        analyzer='word'
    )

    count_vec = CountVectorizer(
        ngram_range=(1, 1),
        max_features=5000,
        min_df=2,
        max_df=0.9
    )

    print("Creating TF-IDF features (unigram + bigram)...")
    X_tfidf1 = tfidf1.fit_transform(train['cleaned_review'])
    X_test_tfidf1 = tfidf1.transform(test['cleaned_review'])

    print("Creating TF-IDF features (bigram + trigram)...")
    X_tfidf2 = tfidf2.fit_transform(train['cleaned_review'])
    X_test_tfidf2 = tfidf2.transform(test['cleaned_review'])

    print("Creating Count features...")
    X_count = count_vec.fit_transform(train['cleaned_review'])
    X_test_count = count_vec.transform(test['cleaned_review'])

    from scipy.sparse import hstack
    X_custom = train[['review_len', 'sentiment_word_count', 'exclamation_count', 'question_count', 'caps_ratio']].values
    X_test_custom = test[['review_len', 'sentiment_word_count', 'exclamation_count', 'question_count', 'caps_ratio']].values

    scaler = StandardScaler()
    X_custom_scaled = scaler.fit_transform(X_custom)
    X_test_custom_scaled = scaler.transform(X_test_custom)

    from scipy.sparse import csr_matrix
    X_custom_sparse = csr_matrix(X_custom_scaled)
    X_test_custom_sparse = csr_matrix(X_test_custom_scaled)

    X_combined = hstack([X_tfidf1, X_tfidf2, X_count, X_custom_sparse])
    X_test_combined = hstack([X_test_tfidf1, X_test_tfidf2, X_test_count, X_test_custom_sparse])

    y = train['sentiment']

    print(f"\nTotal features: {X_combined.shape[1]}")

    print("\n" + "="*60)
    print("Step 3: Training ensemble model")
    print("="*60)

    clf1 = LogisticRegression(max_iter=2000, C=0.5, solver='lbfgs')
    clf2 = LinearSVC(max_iter=2000, C=0.1, loss='squared_hinge')
    clf3 = LogisticRegression(max_iter=2000, C=1.0, solver='saga', penalty='l1')

    ensemble = VotingClassifier(
        estimators=[('lr1', clf1), ('svc', clf2), ('lr2', clf3)],
        voting='soft',
        weights=[2, 1, 1]
    )

    print("\nCross-validation...")
    scores = cross_val_score(ensemble, X_combined, y, cv=5, scoring='roc_auc', n_jobs=1)
    print(f"Cross-validation AUC scores: {scores}")
    print(f"Mean CV AUC: {np.mean(scores):.4f} (std: {np.std(scores):.4f})")

    print("\nTraining ensemble on full dataset...")
    ensemble.fit(X_combined, y)

    print("\n" + "="*60)
    print("Step 4: Making predictions")
    print("="*60)

    test_pred_proba = ensemble.predict_proba(X_test_combined)[:, 1]
    test_pred = (test_pred_proba >= 0.5).astype(int)

    output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    output.to_csv("submission_v3.csv", index=False, quoting=3)

    print(f"\nSubmission file created: submission_v3.csv")
    print(f"Total predictions: {len(output)}")
    print(f"Positive predictions: {(test_pred == 1).sum()}")
    print(f"Negative predictions: {(test_pred == 0).sum()}")

    print("\n" + "="*60)
    print("Sample submission:")
    print("="*60)
    print(output.head(10))

    with open('final_model_v3.pkl', 'wb') as f:
        pickle.dump({
            'tfidf1': tfidf1,
            'tfidf2': tfidf2,
            'count_vec': count_vec,
            'scaler': scaler,
            'ensemble': ensemble,
            'cv_auc': np.mean(scores)
        }, f)

    print("\n" + "="*60)
    print("Version 3 pipeline complete!")
    print("="*60)
    print(f"Cross-validation AUC: {np.mean(scores):.4f}")
    print("Files saved:")
    print("  - submission_v3.csv (submission file)")
    print("  - final_model_v3.pkl (trained models)")

if __name__ == '__main__':
    main()