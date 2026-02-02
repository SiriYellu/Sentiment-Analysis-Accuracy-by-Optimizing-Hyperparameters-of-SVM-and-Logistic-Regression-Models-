import pandas as pd
import numpy as np
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tqdm import tqdm
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentDataProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.nlp = spacy.load('en_core_web_sm')
        
    def clean_text(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        
        doc = self.nlp(text)
        tokens = [token.text.lower() for token in doc if token.is_alpha]
        return ' '.join(tokens)

    def get_sentiment_label(self, text):
        try:
            if not text or len(text.strip()) < 5:
                return 1
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.softmax(outputs.logits, dim=1)
                sentiment_score = torch.argmax(predictions).item()
            
            if sentiment_score in [0, 1]:
                return 0
            elif sentiment_score in [4, 5]:
                return 2
            else:
                return 1
                
        except Exception as e:
            print(f"Error processing text: {str(e)}")
            return 1

    def process_datasets(self, twitter_path, linkedin_path):
        print("Loading datasets...")
        twitter_df = pd.read_csv(twitter_path)
        linkedin_df = pd.read_csv(linkedin_path)
        
        print("Cleaning texts...")
        twitter_df['cleaned_text'] = twitter_df['text'].apply(self.clean_text)
        linkedin_df['cleaned_text'] = linkedin_df['text'].apply(self.clean_text)
        
        twitter_df = twitter_df[twitter_df['cleaned_text'].str.len() > 5].reset_index(drop=True)
        linkedin_df = linkedin_df[linkedin_df['cleaned_text'].str.len() > 5].reset_index(drop=True)
        
        print("Generating sentiment labels for Twitter dataset...")
        twitter_df['sentiment'] = [
            self.get_sentiment_label(text) 
            for text in tqdm(twitter_df['cleaned_text'])
        ]
        
        print("Generating sentiment labels for LinkedIn dataset...")
        linkedin_df['sentiment'] = [
            self.get_sentiment_label(text) 
            for text in tqdm(linkedin_df['cleaned_text'])
        ]
        
        twitter_final = twitter_df[['cleaned_text', 'sentiment']]
        linkedin_final = linkedin_df[['cleaned_text', 'sentiment']]
        
        twitter_final.to_csv('processed_twitter.csv', index=False)
        linkedin_final.to_csv('processed_linkedin.csv', index=False)
        
        print("\nDataset Statistics:")
        print("\nTwitter Dataset:")
        print(f"Total samples: {len(twitter_final)}")
        print("Sentiment distribution:")
        print(twitter_final['sentiment'].value_counts(normalize=True))
        
        print("\nLinkedIn Dataset:")
        print(f"Total samples: {len(linkedin_final)}")
        print("Sentiment distribution:")
        print(linkedin_final['sentiment'].value_counts(normalize=True))
        
        return twitter_final, linkedin_final

class ModelEvaluator:
    def __init__(self, max_len=128, batch_size=32):
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.results = {}
        
    def prepare_data(self, twitter_data, linkedin_data):
        """Prepare data for model training and evaluation."""
        print("Vectorizing text data...")
        self.vectorizer = TfidfVectorizer(max_features=5000)
        
        X_train = self.vectorizer.fit_transform(twitter_data['cleaned_text'])
        y_train = twitter_data['sentiment']
        
        X_val = self.vectorizer.transform(linkedin_data['cleaned_text'])
        y_val = linkedin_data['sentiment']
        
        return X_train, y_train, X_val, y_val
    
    def train_traditional_models(self, X_train, y_train, X_val, y_val):
        """Train and evaluate traditional ML models."""
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'SVM': LinearSVC(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        }
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            accuracy = accuracy_score(y_val, y_pred)
            report = classification_report(y_val, y_pred, output_dict=True)
            
            self.models[name] = model
            self.results[name] = {
                'accuracy': accuracy,
                'report': report
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_val, y_pred))
    
    def train_bert(self, twitter_data, linkedin_data, epochs=4):
        """Train and evaluate BERT model."""
        print("\nTraining BERT model...")
        
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3
        ).to(self.device)
        
        # Prepare datasets
        train_encodings = tokenizer(
            list(twitter_data['cleaned_text']),
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        val_encodings = tokenizer(
            list(linkedin_data['cleaned_text']),
            truncation=True,
            padding=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.tensor(twitter_data['sentiment'].values)
        )
        
        val_dataset = TensorDataset(
            val_encodings['input_ids'],
            val_encodings['attention_mask'],
            torch.tensor(linkedin_data['sentiment'].values)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                batch_input_ids, batch_attention_mask, batch_labels = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                outputs = model(
                    batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                batch_input_ids, batch_attention_mask, batch_labels = [b.to(self.device) for b in batch]
                
                outputs = model(
                    batch_input_ids,
                    attention_mask=batch_attention_mask
                )
                
                predictions.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                true_labels.extend(batch_labels.cpu().numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        
        self.models['BERT'] = model
        self.results['BERT'] = {
            'accuracy': accuracy,
            'report': report
        }
        
        print(f"\nBERT Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
    
    def plot_results(self):
        """Plot comparison of model performances."""
        accuracies = {name: results['accuracy'] for name, results in self.results.items()}
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(accuracies.keys(), accuracies.values())
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()
        
        # Save detailed results
        with open('model_results.txt', 'w') as f:
            f.write("Detailed Model Results\n\n")
            for name, results in self.results.items():
                f.write(f"\n{name}\n")
                f.write(f"Accuracy: {results['accuracy']:.4f}\n")
                f.write("Classification Report:\n")
                f.write(str(classification_report(None, None, output_dict=False, target_names=['Negative', 'Neutral', 'Positive'])))
                f.write("\n" + "="*50 + "\n")

def main():
    print("Initializing Sentiment Data Processor...")
    data_processor = SentimentDataProcessor()
    
    print("Processing datasets...")
    twitter_data, linkedin_data = data_processor.process_datasets(
        'vaccination_tweets.csv',
        'linkedin_temp.csv'
    )
    
    print("\nInitializing Model Evaluator...")
    evaluator = ModelEvaluator()
    
    print("Preparing data for model training...")
    X_train, y_train, X_val, y_val = evaluator.prepare_data(twitter_data, linkedin_data)
    
    print("Training and evaluating models...")
    evaluator.train_traditional_models(X_train, y_train, X_val, y_val)
    evaluator.train_bert(twitter_data, linkedin_data, epochs=4)
    
    print("Plotting results...")
    evaluator.plot_results()
    
    print("\nAll results have been saved to 'model_results.txt' and 'model_comparison.png'")

if __name__ == "__main__":
    main()