import pandas as pd
import numpy as np
import re
import pickle
import logging
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import nltk

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

def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

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
    unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    print(f"Read {train['review'].size} labeled train reviews, {test['review'].size} labeled test reviews, and {unlabeled_train['review'].size} unlabeled reviews\n")

    print("Loading tokenizer...")
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    print("Parsing sentences from training set...")
    sentences = []
    for i, review in enumerate(train["review"]):
        sentences += review_to_sentences(review, tokenizer)
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i + 1} / {len(train)} training reviews")

    print(f"Training set sentences: {len(sentences)}")

    print("Parsing sentences from unlabeled set...")
    for i, review in enumerate(unlabeled_train["review"]):
        sentences += review_to_sentences(review, tokenizer)
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i + 1} / {len(unlabeled_train)} unlabeled reviews")

    print(f"Total sentences: {len(sentences)}")

    num_features = 300
    min_word_count = 40
    num_workers = 4
    context = 10
    downsampling = 1e-3

    print(f"\nTraining Word2Vec model with:")
    print(f"  - num_features: {num_features}")
    print(f"  - min_word_count: {min_word_count}")
    print(f"  - num_workers: {num_workers}")
    print(f"  - context: {context}")
    print(f"  - downsampling: {downsampling}")

    from gensim.models import word2vec
    print("\nTraining model...")
    model = word2vec.Word2Vec(
        sentences,
        workers=num_workers,
        vector_size=num_features,
        min_count=min_word_count,
        window=context,
        sample=downsampling
    )

    model.init_sims(replace=True)

    model_name = "300features_40minwords_10context"
    model.save(model_name)
    print(f"\nModel saved as '{model_name}'")

    print("\n" + "="*50)
    print("Exploring model results:")
    print("="*50)

    print("\n1. Testing doesnt_match:")
    try:
        print(f"   'man woman child kitchen' -> {model.wv.doesnt_match('man woman child kitchen'.split())}")
        print(f"   'france england germany berlin' -> {model.wv.doesnt_match('france england germany berlin'.split())}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n2. Testing most_similar for 'man':")
    try:
        similar = model.wv.most_similar("man")
        for word, score in similar[:5]:
            print(f"   {word}: {score:.4f}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n3. Testing most_similar for 'awful':")
    try:
        similar = model.wv.most_similar("awful")
        for word, score in similar[:5]:
            print(f"   {word}: {score:.4f}")
    except Exception as e:
        print(f"   Error: {e}")

    print("\n4. Vocabulary size:", len(model.wv))

    with open('word2vec_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'num_features': num_features,
            'min_word_count': min_word_count,
            'context': context
        }, f)

    print("\nWord2Vec model and parameters saved to 'word2vec_model.pkl'")

if __name__ == '__main__':
    main()