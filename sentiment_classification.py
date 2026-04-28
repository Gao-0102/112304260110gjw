import pandas as pd
import numpy as np
import re
import pickle
import logging
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
from gensim.models import word2vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review, "html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    return words

def review_to_sentences(review, tokenizer, remove_stopwords=False):
    raw_sentences = tokenizer.tokenize(review.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(review_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            nwords += 1
            featureVec = np.add(featureVec, model.wv[word])
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        if counter % 1000 == 0:
            print(f"  Processing review {counter} of {len(reviews)}")
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return reviewFeatureVecs

def main():
    print("="*60)
    print("Step 1: Loading and cleaning data")
    print("="*60)

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

    print("Loading data...")
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

    print(f"Train size: {len(train)}, Test size: {len(test)}")

    print("\nCleaning training reviews...")
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    clean_train_reviews = []
    for i, review in enumerate(train["review"]):
        clean_train_reviews.append(review_to_wordlist(review))
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1} / {len(train)} training reviews")

    print("\nCleaning test reviews...")
    clean_test_reviews = []
    for i, review in enumerate(test["review"]):
        clean_test_reviews.append(review_to_wordlist(review))
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1} / {len(test)} test reviews")

    print("\n" + "="*60)
    print("Step 2: Training Word2Vec model")
    print("="*60)

    sentences = []
    print("Parsing sentences from training set...")
    for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    print("Parsing sentences from unlabeled set...")
    unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review, tokenizer)

    print(f"Total sentences: {len(sentences)}")

    num_features = 300
    min_word_count = 40
    num_workers = 4
    context = 10
    downsampling = 1e-3

    print(f"\nTraining Word2Vec model...")
    model = word2vec.Word2Vec(
        sentences,
        workers=num_workers,
        vector_size=num_features,
        min_count=min_word_count,
        window=context,
        sample=downsampling
    )
    model.init_sims(replace=True)
    print(f"Word2Vec model trained. Vocabulary size: {len(model.wv)}")

    print("\n" + "="*60)
    print("Step 3: Creating average word embeddings")
    print("="*60)

    print("\nCreating averaged feature vectors for training set...")
    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

    print("\nCreating averaged feature vectors for test set...")
    testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

    print("\n" + "="*60)
    print("Step 4: Training Logistic Regression")
    print("="*60)

    X = trainDataVecs
    y = train["sentiment"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, y_pred_proba)
    print(f"\nValidation ROC AUC Score: {auc_score:.4f}")

    print("\nRetraining on full training set...")
    clf_final = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    clf_final.fit(X, y)

    print("\n" + "="*60)
    print("Step 5: Making predictions and creating submission file")
    print("="*60)

    test_pred = clf_final.predict(testDataVecs)

    output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    output.to_csv("submission.csv", index=False, quoting=3)

    print(f"\nSubmission file created: submission.csv")
    print(f"Total predictions: {len(output)}")
    print(f"Positive predictions: {(test_pred == 1).sum()}")
    print(f"Negative predictions: {(test_pred == 0).sum()}")

    print("\n" + "="*60)
    print("Sample submission:")
    print("="*60)
    print(output.head(10))

    with open('final_model.pkl', 'wb') as f:
        pickle.dump({
            'word2vec_model': model,
            'logistic_regression': clf_final,
            'num_features': num_features,
            'auc_score': auc_score
        }, f)

    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)
    print(f"Validation AUC: {auc_score:.4f}")
    print("Files saved:")
    print("  - submission.csv (submission file)")
    print("  - final_model.pkl (trained models)")

if __name__ == '__main__':
    main()