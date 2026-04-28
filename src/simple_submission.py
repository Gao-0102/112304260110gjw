import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)

def clean_text(raw_text):
    review_text = BeautifulSoup(raw_text, "html.parser").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    negation_words = {'not', 'no', 'never', 'nor', 'none', 'hardly', 'scarcely', 'barely', 'cannot'}
    words = [w for w in words if w not in stops or w in negation_words]
    return " ".join(words)

print("Cleaning training data...")
train_cleaned = [clean_text(text) for text in train["review"]]
print("Cleaning test data...")
test_cleaned = [clean_text(text) for text in test["review"]]

print("\nExtracting features...")
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=20000,
    sublinear_tf=True
)

X_train = tfidf.fit_transform(train_cleaned)
X_test = tfidf.transform(test_cleaned)
y_train = train["sentiment"]

print(f"Feature shape: {X_train.shape}")

print("\nTraining Logistic Regression...")
lr = LogisticRegression(C=10, solver='liblinear', max_iter=1000)

print("Cross-validation...")
scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='roc_auc')
print(f"Cross-validation AUC: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

print("\nTraining on full dataset...")
lr.fit(X_train, y_train)

result = lr.predict_proba(X_test)[:, 1]

output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
output.to_csv("submission.csv", index=False, quoting=3)

print("\nSubmission file created: submission.csv")
print(f"Predictions range: {result.min():.4f} - {result.max():.4f}")
print(f"Mean prediction: {result.mean():.4f}")
print("\nSample output (first 10 rows):")
print(output.head(10))

print("\n" + "="*60)
print("Top 20 important features (positive sentiment):")
print("="*60)
feature_names = tfidf.get_feature_names_out()
coef = lr.coef_[0]
top_positive = coef.argsort()[-20:][::-1]
for i in top_positive:
    print(f"  {feature_names[i]:<25} {coef[i]:.4f}")

print("\n" + "="*60)
print("Top 20 important features (negative sentiment):")
print("="*60)
top_negative = coef.argsort()[:20]
for i in top_negative:
    print(f"  {feature_names[i]:<25} {coef[i]:.4f}")