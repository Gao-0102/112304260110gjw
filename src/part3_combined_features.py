import pandas as pd
import re
import os
import random
from bs4 import BeautifulSoup
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.sparse import hstack
import joblib

RANDOM_SEED = 1993
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# 自定义停用词列表，保留否定词
def get_custom_stopwords():
    # 常见英文停用词列表
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what', 'which', 'this',
        'that', 'these', 'those', 'then', 'just', 'so', 'than', 'such', 'both', 'through',
        'about', 'for', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
        'had', 'having', 'do', 'does', 'did', 'doing', 'to', 'from', 'up', 'down', 'in',
        'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'don', 'should', 'now'
    }
    # 保留否定词
    negation_words = {'not', 'no', 'never', 'nor', 'nothing', 'nowhere', 'none', 'nobody', 'noone'}
    stop_words = stop_words - negation_words
    return stop_words

custom_stopwords = get_custom_stopwords()

# 简单的词干提取函数
def simple_stem(word):
    # 简单的词干提取规则
    suffixes = ['ing', 'ed', 'es', 's']
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

# 改进的文本清理函数
def clean_review(text: str) -> str:
    # 1. 去 HTML 标签
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # 2. 保留标点符号，但处理特殊情况
    # 保留情感相关的标点，如感叹号、问号
    # 处理缩写形式，如 don't -> dont
    text = text.replace("'t", "t")
    
    # 3. 小写化
    text = text.lower()
    
    # 4. 分词
    words = re.findall(r'\b\w+\b|[!\?]+', text)
    
    # 5. 简单词干提取
    words = [simple_stem(word) for word in words]
    
    # 6. 移除停用词（保留否定词）
    words = [word for word in words if word not in custom_stopwords]
    
    # 7. 重新组合文本
    text = ' '.join(words)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# NB-SVM log-count ratio
def nbsvm_ratio(X, y, alpha=1.0):
    y = np.asarray(y)
    pos = X[y == 1].sum(axis=0) + alpha
    neg = X[y == 0].sum(axis=0) + alpha
    pos = pos / pos.sum()
    neg = neg / neg.sum()
    r = np.log(pos / neg)
    return np.asarray(r).ravel()  # 1D

# 读取训练数据
train = pd.read_csv('labeledTrainData.tsv', header=0, delimiter='\t', quoting=3)
test = pd.read_csv('testData.tsv', header=0, delimiter='\t', quoting=3)

# 处理数据
print("Processing training data...")
train_text = train["review"].apply(clean_review).tolist()
test_text = test["review"].apply(clean_review).tolist()
y = train["sentiment"].values

all_text = train_text + test_text

# Word + Character TF-IDF features
print("Building TF-IDF features (word + char)...")

word_vec = TfidfVectorizer(
    min_df=3,
    max_df=0.9,
    max_features=40000,
    ngram_range=(1, 2),
    sublinear_tf=True,
)

char_vec = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=3,
    max_features=40000,
    sublinear_tf=True,
)

X_word_all = word_vec.fit_transform(all_text)
X_char_all = char_vec.fit_transform(all_text)

X_all = hstack([X_word_all, X_char_all]).tocsr()

X = X_all[: len(train_text)]
X_test = X_all[len(train_text) :]

print("X shape:", X.shape, "X_test shape:", X_test.shape)

# 7-fold CV
N_FOLDS = 7
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

oof = np.zeros(len(y))
fold_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n========== FOLD {fold}/{N_FOLDS} ==========")
    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    # ----- Model 1: plain LR -----
    lr1 = LogisticRegression(
        C=4.0,
        max_iter=800,   # "more epochs"
        solver="liblinear",
        n_jobs=-1,
    )
    lr1.fit(X_tr, y_tr)
    p1 = lr1.predict_proba(X_va)[:, 1]

    # ----- Model 2: NB-SVM style LR -----
    r = nbsvm_ratio(X_tr, y_tr, alpha=1.0)
    X_tr_nb = X_tr.multiply(r)
    X_va_nb = X_va.multiply(r)

    lr2 = LogisticRegression(
        C=4.0,
        max_iter=800,   # "more epochs"
        solver="liblinear",
        n_jobs=-1,
    )
    lr2.fit(X_tr_nb, y_tr)
    p2 = lr2.predict_proba(X_va_nb)[:, 1]

    # ----- Blend -----
    p = 0.5 * p1 + 0.5 * p2
    oof[va_idx] = p

    fold_auc = roc_auc_score(y_va, p)
    fold_scores.append(fold_auc)
    print(f"Fold {fold} AUC: {fold_auc:.5f}")

cv_auc = roc_auc_score(y, oof)
print("\n==============================")
print("OOF CV AUC:", cv_auc)
print("Fold AUCs:", [round(s, 5) for s in fold_scores])
print("==============================")

# Train final models on all data
print("\nTraining final models on FULL data...")
final_lr1 = LogisticRegression(
    C=4.0,
    max_iter=1000,   # even more iterations for final model
    solver="liblinear",
    n_jobs=-1,
)
final_lr1.fit(X, y)

r_full = nbsvm_ratio(X, y, alpha=1.0)
X_nb_full = X.multiply(r_full)
X_test_nb = X_test.multiply(r_full)

final_lr2 = LogisticRegression(
    C=4.0,
    max_iter=1000,
    solver="liblinear",
    n_jobs=-1,
)
final_lr2.fit(X_nb_full, y)

# Predict test & save submission
print("Predicting test set...")
test_p1 = final_lr1.predict_proba(X_test)[:, 1]
test_p2 = final_lr2.predict_proba(X_test_nb)[:, 1]
test_pred = 0.5 * test_p1 + 0.5 * test_p2

# 转换为二分类结果
test_pred_class = (test_pred >= 0.5).astype(int)

# 生成提交文件
submission = pd.DataFrame({
    "id": test["id"],
    "sentiment": test_pred_class,
})

# 确保所有字段都没有引号
submission.to_csv("submission.csv", index=False, quoting=3)
print("Saved submission.csv")

# 保存模型
joblib.dump(final_lr1, 'final_lr1_model.pkl')
joblib.dump(final_lr2, 'final_lr2_model.pkl')
joblib.dump(word_vec, 'word_vectorizer.pkl')
joblib.dump(char_vec, 'char_vectorizer.pkl')
print("\nModels saved successfully")
