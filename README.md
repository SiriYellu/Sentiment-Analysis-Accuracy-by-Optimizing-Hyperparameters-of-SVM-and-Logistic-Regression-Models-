# Sentiment-Analysis-Accuracy-by-Optimizing-Hyperparameters-of-SVM-and-Logistic-Regression-Models
# Sentiment Analysis Pipeline: Twitter vs LinkedIn

## Overview
This project implements an end-to-end sentiment analysis pipeline to process social media text data and compare the performance of traditional machine learning models with a transformer-based deep learning model (BERT). Twitter data is used for training, while LinkedIn data is used for cross-platform validation.

The pipeline includes:
- Text cleaning and preprocessing
- Automatic sentiment labeling using a pretrained transformer
- Feature extraction using TF-IDF
- Model training and evaluation
- Performance comparison and visualization

---

## Datasets
- **Twitter Dataset**
  - Size: 10,869 samples
  - Sentiment Distribution:
    - Positive: 40.38%
    - Negative: 35.16%
    - Neutral: 24.45%

- **LinkedIn Dataset**
  - Size: 277 samples
  - Sentiment Distribution:
    - Positive: 35.38%
    - Negative: 34.66%
    - Neutral: 29.96%

Twitter data is used for model training, while LinkedIn data serves as a validation set to test generalization across platforms.

---

## Libraries and Tools
- **Data Processing:** pandas, numpy, regex
- **NLP:** spaCy, Hugging Face Transformers
- **Machine Learning:** scikit-learn, XGBoost
- **Deep Learning:** PyTorch
- **Visualization:** matplotlib, seaborn
- **Utilities:** tqdm

---

## Pipeline Architecture

### 1. Data Cleaning
- Removes URLs, mentions, and hashtags
- Tokenizes text using spaCy
- Retains only alphabetic tokens
- Converts text to lowercase

### 2. Sentiment Label Generation
- Uses `nlptown/bert-base-multilingual-uncased-sentiment`
- Converts 5-class predictions into 3 classes:
  - **0:** Negative
  - **1:** Neutral
  - **2:** Positive
- Filters out very short or invalid texts

### 3. Feature Engineering
- TF-IDF vectorization
- Maximum features: 5,000
- Sparse representation for traditional models

---

## Models Evaluated

### Traditional Machine Learning Models
- Logistic Regression
- Support Vector Machine (LinearSVC)
- Random Forest
- XGBoost

### Transformer-Based Model
- BERT (`bert-base-uncased`)
- Fine-tuned on Twitter data
- 3-class classification head
- Trained for 4 epochs using AdamW optimizer

---

## Model Performance

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | 54.15%   |
| SVM                | 56.68%   |
| Random Forest      | 46.93%   |
| XGBoost            | 47.65%   |
| **BERT**           | **74.37%** |
<img width="1000" height="600" alt="model_comparison" src="https://github.com/user-attachments/assets/edeebe8a-d7e4-4963-81f9-8f12d9c9570d" />

---

## Key Observations
- Linear models (Logistic Regression, SVM) outperform ensemble models on sparse text features.
- Random Forest and XGBoost struggle with high-dimensional TF-IDF representations.
- BERT significantly outperforms all traditional models due to:
  - Contextual word embeddings
  - Ability to capture semantic and syntactic dependencies
  - Robustness to noisy social media text
- Neutral sentiment remains the most challenging class across traditional models.

---

## Results and Outputs
The pipeline generates the following artifacts:
- `processed_twitter.csv` – Cleaned and labeled Twitter data
- `processed_linkedin.csv` – Cleaned and labeled LinkedIn data
- `model_results.txt` – Detailed classification reports
- `model_comparison.png` – Accuracy comparison plot

---

## Conclusion
This study demonstrates the clear advantage of transformer-based models for sentiment analysis tasks involving complex, noisy, and cross-platform text data. While traditional machine learning models provide reasonable baselines, BERT achieves substantially higher accuracy and balanced performance across sentiment classes.

Future improvements may include:
- Hyperparameter tuning
- Class imbalance handling
- Domain-adaptive pretraining
- Data augmentation techniques

---

## Author
**Siri Yellu**  
MS in Computer Science (Data Science Concentration)  
Kennesaw State University
