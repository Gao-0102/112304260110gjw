# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息
- **姓名**：
- **学号**：
- **班级**：

> 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验任务
本实验基于给定文本数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：
- 文本预处理
- Word2Vec 词向量训练或加载
- 句子向量表示
- 分类模型训练
- Kaggle 结果提交与分析

---

## 3. 比赛与提交信息
- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：https://www.kaggle.com/competitions/word2vec-nlp-tutorial/overview
- **提交日期**：2026-04-16

- **GitHub 仓库地址**：
- **GitHub README 地址**：

> 注意：GitHub 仓库首页或 README 页面中，必须能看到“姓名 + 学号”，否则无效。

---

## 4. Kaggle 成绩
请填写你最终提交到 Kaggle 的结果：

- **Public Score**：
- **Private Score**（如有）：
- **排名**（如能看到可填写）：

---

## 5. Kaggle 截图
请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。  
> 截图文件名示例：`2023123456_张三_kaggle_score.png`

---

## 6. 实验方法说明

### （1）文本预处理
请说明你对文本做了哪些处理，例如：
- 分词
- 去停用词
- 去除标点或特殊符号
- 转小写

**我的做法：**  
- 移除 HTML 标签
- 处理缩写形式（如 don't -> do not）
- 转为小写
- 保留情感相关的标点（感叹号、问号）
- 移除停用词，但保留否定词（not, no, never等）
- 简单词干提取

---

### （2）Word2Vec 特征表示
请说明你如何使用 Word2Vec，例如：
- 是自己训练 Word2Vec，还是使用已有模型
- 词向量维度是多少
- 句子向量如何得到（平均、加权平均、池化等）

**我的做法：**  
- 使用自己训练的 Word2Vec 模型
- 词向量维度为 300
- 句子向量通过对文本中所有词的词向量取平均得到

---

### （3）分类模型
请说明你使用了什么分类模型，例如：
- Logistic Regression
- Random Forest
- SVM
- XGBoost

并说明最终采用了哪一个模型。

**我的做法：**  
- 双 TF-IDF + 逻辑回归模型（使用短语模式）
- Word2Vec + 逻辑回归模型
- 最终采用了模型集成（平均融合）

---

## 7. 实验流程
请简要说明你的实验流程。

示例：
1. 读取训练集和测试集  
2. 对文本进行预处理  
3. 训练或加载 Word2Vec 模型  
4. 将每条文本表示为句向量  
5. 用训练集训练分类器  
6. 在测试集上预测结果  
7. 生成 submission 文件并提交 Kaggle  

**我的实验流程：**  
1. 读取训练集和测试集
2. 对文本进行预处理（去HTML标签、处理缩写、小写化、保留否定词等）
3. 训练 Word2Vec 模型
4. 提取双 TF-IDF 特征（词级+字符级，使用短语模式）
5. 提取 Word2Vec 特征
6. 训练多个逻辑回归模型
7. 7折分层交叉验证评估模型性能
8. 模型集成（平均融合）
9. 生成 submission 文件并提交 Kaggle

---

## 8. 文件说明
请说明仓库中各文件或文件夹的作用。

示例：
- `data/`：存放数据文件
- `src/`：存放源代码
- `notebooks/`：存放实验 notebook
- `images/`：存放 README 中使用的图片
- `submission/`：存放提交文件

**我的项目结构：**
```text
Bag_Popcorn/
├─ __pycache__/：Python 编译缓存
├─ labeledTrainData.tsv：训练数据
├─ testData.tsv：测试数据
├─ unlabeledTrainData.tsv：无标签训练数据
├─ part1_bag_of_words.py：词袋模型实现
├─ part2_word2vec.py：Word2Vec 模型实现
├─ part3_combined_features.py：组合特征实现
├─ part3_word2vec_logistic.py：Word2Vec + 逻辑回归实现
├─ part3_word2vec_sentiment.py：Word2Vec + 情感分析实现
├─ part3_word2vec_svr.py：Word2Vec + SVR 实现
├─ part3_word2vec_xgb.py：Word2Vec + XGBoost 实现
├─ model_ensemble.py：模型集成实现
├─ word2vec_model.bin：训练好的 Word2Vec 模型
├─ *.pkl：训练好的分类模型
├─ submission.csv：最终提交文件
├─ requirements.txt：Python 依赖
├─ .env.example：环境变量模板
├─ .gitignore：Git 忽略文件
└─ README.md：项目说明


