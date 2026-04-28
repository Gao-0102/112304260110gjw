import pandas as pd
import numpy as np
import re
import pickle
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return " ".join(meaningful_words)

def main():
    print("Loading labeled training data...")
    labeled_train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    
    print("Loading unlabeled training data...")
    unlabeled_train = pd.read_csv('unlabeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    
    print("Loading test data...")
    test_data = pd.read_csv('testData.tsv', header=0, delimiter="\t", quoting=3)
    
    print(f"Labeled train shape: {labeled_train.shape}")
    print(f"Unlabeled train shape: {unlabeled_train.shape}")
    print(f"Test shape: {test_data.shape}")
    
    print("\nSample raw review:")
    print(labeled_train["review"][0][:500] + "...")
    
    print("\nPreprocessing labeled training data...")
    num_reviews = labeled_train["review"].size
    labeled_train['cleaned_review'] = labeled_train['review'].apply(review_to_words)
    
    print("Preprocessing unlabeled training data...")
    unlabeled_train['cleaned_review'] = unlabeled_train['review'].apply(review_to_words)
    
    print("Preprocessing test data...")
    test_data['cleaned_review'] = test_data['review'].apply(review_to_words)
    
    print("\nSample cleaned review:")
    print(labeled_train["cleaned_review"][0][:200] + "...")
    
    print("\nSaving processed data...")
    labeled_train.to_csv('labeledTrainData_processed.tsv', sep='\t', index=False, quoting=3)
    unlabeled_train.to_csv('unlabeledTrainData_processed.tsv', sep='\t', index=False, quoting=3)
    test_data.to_csv('testData_processed.tsv', sep='\t', index=False, quoting=3)
    
    with open('preprocessed_data.pkl', 'wb') as f:
        pickle.dump({
            'labeled_train': labeled_train,
            'unlabeled_train': unlabeled_train,
            'test_data': test_data
        }, f)
    
    print("\nPreprocessing complete!")
    print("Processed files saved:")
    print("- labeledTrainData_processed.tsv")
    print("- unlabeledTrainData_processed.tsv")
    print("- testData_processed.tsv")
    print("- preprocessed_data.pkl")

if __name__ == '__main__':
    main()