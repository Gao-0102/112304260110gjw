import pandas as pd
import numpy as np
import re
import pickle
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def review_to_wordlist(review, remove_stopwords=True):
    review_text = BeautifulSoup(review, "html.parser").get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        negation_words = {'not', 'no', 'never', 'nor', 'none', 'hardly', 'scarcely', 'barely', 'cannot'}
        words = [w for w in words if w not in stops or w in negation_words]
    
    return " ".join(words)

def main():
    print("="*60)
    print("Step 1: Loading and cleaning data")
    print("="*60)

    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')

    print("Loading data...")
    train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

    print(f"Train size: {len(train)}, Test size: {len(test)}")

    print("\nCleaning training reviews...")
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
    print("Step 2: TF-IDF with Bigram")
    print("="*60)

    print("\nCreating TF-IDF features with Bigram...")
    tfidf = TfidfVectorizer(
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 2),
        max_features=5000,
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )

    X = tfidf.fit_transform(clean_train_reviews)
    X_test = tfidf.transform(clean_test_reviews)
    y = train["sentiment"]

    print(f"TF-IDF feature shape: {X.shape}")
    print(f"Number of features (unigrams + bigrams): {X.shape[1]}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n" + "="*60)
    print("Step 3: Training models")
    print("="*60)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs'),
        'Ridge Classifier': RidgeClassifier(alpha=1.0),
        'Linear SVC': LinearSVC(max_iter=2000, C=1.0)
    }

    best_model = None
    best_auc = 0.0
    best_model_name = ""

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_val)
        
        auc_score = roc_auc_score(y_val, y_pred_proba)
        print(f"  Validation ROC AUC: {auc_score:.4f}")
        
        if auc_score > best_auc:
            best_auc = auc_score
            best_model = model
            best_model_name = name

    print(f"\nBest model: {best_model_name} with AUC: {best_auc:.4f}")

    print("\nRetraining best model on full training set...")
    best_model.fit(X, y)

    print("\n" + "="*60)
    print("Step 4: Making predictions")
    print("="*60)

    if hasattr(best_model, 'predict_proba'):
        test_pred_proba = best_model.predict_proba(X_test)[:, 1]
    else:
        test_pred_proba = best_model.decision_function(X_test)
    
    test_pred = (test_pred_proba >= 0.5).astype(int)

    output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    output.to_csv("submission_improved.csv", index=False, quoting=3)

    print(f"\nSubmission file created: submission_improved.csv")
    print(f"Total predictions: {len(output)}")
    print(f"Positive predictions: {(test_pred == 1).sum()}")
    print(f"Negative predictions: {(test_pred == 0).sum()}")

    print("\n" + "="*60)
    print("Sample submission:")
    print("="*60)
    print(output.head(10))

    with open('final_model_improved.pkl', 'wb') as f:
        pickle.dump({
            'tfidf': tfidf,
            'model': best_model,
            'model_name': best_model_name,
            'auc_score': best_auc
        }, f)

    print("\n" + "="*60)
    print("Improved pipeline complete!")
    print("="*60)
    print(f"Best Model: {best_model_name}")
    print(f"Validation AUC: {best_auc:.4f}")
    print("Files saved:")
    print("  - submission_improved.csv (submission file)")
    print("  - final_model_improved.pkl (trained models)")

    print("\n" + "="*60)
    print("Top 20 features (bigrams + unigrams):")
    print("="*60)
    feature_names = np.array(tfidf.get_feature_names_out())
    if hasattr(best_model, 'coef_'):
        coefs = best_model.coef_[0]
        top_positive = np.argsort(coefs)[-10:]
        top_negative = np.argsort(coefs)[:10]
        
        print("\nPositive sentiment features:")
        for i in reversed(top_positive):
            print(f"  {feature_names[i]:<20} {coefs[i]:.4f}")
        
        print("\nNegative sentiment features:")
        for i in top_negative:
            print(f"  {feature_names[i]:<20} {coefs[i]:.4f}")

if __name__ == '__main__':
    main()