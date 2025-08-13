Toxic-Comment-Classification
📌 Overview

This project implements a **multi-label text classification model** to detect different types of toxic comments such as:

* Toxic
* Severe Toxic
* Obscene
* Threat
* Insult
* Identity Hate

It is built using the **Jigsaw Toxic Comment Classification Challenge** dataset and applies NLP techniques for data cleaning, feature extraction, and classification.

📂 Dataset

The dataset is from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge), containing user comments from Wikipedia discussions labeled for multiple categories of toxicity.

Files used:

* `train.csv` – Training data with labeled comments
* `test.csv` – Test data without labels
* `sample_submission.csv` – Sample output format

🛠️ Methodology

1. Data Preprocessing

* Lowercasing text
* Removing punctuation, numbers, and special characters
* Expanding contractions (e.g., `"can't"` → `"cannot"`)
* Removing stopwords
* Lemmatization/Stemming for word normalization

2. Feature Extraction

TF-IDF Vectorization (word-level, n-grams up to trigrams)

3. Model Training

* One-vs-Rest Logistic Regression for multi-label classification
* Cross-validation to evaluate performance

4. Model Evaluation

* Accuracy, Precision, Recall, and F1-score per label
* ROC-AUC score for overall performance
* Confusion Matrix visualization

5. Prediction Function

* A helper function to classify new comments as Toxic or Non-Toxic

📊 Results

| Label                       | ROC-AUC Score |
| --------------------------- | ------------- |
| Toxic                       | 0.9716        |
| Severe Toxic                | 0.9856        |
| Obscene                     | 0.9844        |
| Threat                      | 0.9792        |
| Insult                      | 0.9771        |
| Identity Hate               | 0.9772        |
| **Average ROC-AUC:** 0.9792 |               |


📦 Requirements

* Python 3.x
* pandas, numpy
* scikit-learn
* nltk
* seaborn, matplotlib

Install dependencies:

```bash
pip install pandas numpy scikit-learn nltk seaborn matplotlib
```

🚀 How to Run

1. Clone this repository

```bash
git clone https://github.com/yourusername/toxic-comment-classification.git
cd toxic-comment-classification
```

2. Download the dataset from [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) and place it in a `/data` folder.
3. Run the notebook or Python script to train and evaluate the model.
4. Use the `make_test_predictions()` function to classify your own text.

📌 Example

```python
comment_text = "I don't think so"
result = make_test_predictions(pd.DataFrame({'id':[1],'comment_text':[comment_text]}), classifier)
print(result)  # Output: NonToxic Comment
```
