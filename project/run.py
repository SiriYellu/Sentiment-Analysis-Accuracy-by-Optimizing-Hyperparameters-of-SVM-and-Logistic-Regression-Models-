import pandas as pd
import numpy as np
import torch
import re
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob

class VaccinationSentimentAnalyzer:
    def __init__(self, max_len=128, batch_size=16):
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = None
        
    def clean_tweet(self, tweet):
        """Clean tweet text by removing URLs, mentions, hashtags, and special characters."""
        tweet = re.sub(r'http\S+', '', str(tweet))
        tweet = re.sub(r'@\w+', '', tweet)
        tweet = re.sub(r'#\w+', '', tweet)
        tweet = re.sub(r'[^\w\s]', '', tweet)
        return tweet.lower().strip()
    
    def get_sentiment_label(self, text):
        """Get sentiment label using TextBlob."""
        analysis = TextBlob(text)
        # Convert polarity to three classes: negative (0), neutral (1), positive (2)
        if analysis.sentiment.polarity < -0.1:
            return 0
        elif analysis.sentiment.polarity > 0.1:
            return 2
        else:
            return 1
    
    def prepare_data(self, data_path):
        """Load and prepare the dataset for training."""
        # Load dataset
        print("Loading dataset...")
        df = pd.read_csv(data_path)
        
        # Basic data preprocessing
        print("Cleaning tweets...")
        df['clean_text'] = df['text'].apply(self.clean_tweet)
        
        # Generate sentiment labels using TextBlob
        print("Generating sentiment labels...")
        df['generated_sentiment'] = df['clean_text'].apply(self.get_sentiment_label)
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Total tweets: {len(df)}")
        print("\nSentiment Distribution:")
        print(df['generated_sentiment'].value_counts(normalize=True).round(3))
        
        # Tokenize tweets
        print("Tokenizing tweets...")
        df['tokens'] = df['clean_text'].apply(
            lambda x: self.tokenizer.encode(x, add_special_tokens=True)
        )
        
        # Pad sequences
        print("Padding sequences...")
        input_ids = pad_sequences(
            df['tokens'].values,
            maxlen=self.max_len,
            dtype="long",
            truncating="post",
            padding="post"
        )
        
        # Create attention masks
        print("Creating attention masks...")
        attention_masks = [[float(i > 0) for i in seq] for seq in input_ids]
        attention_masks = np.array(attention_masks)
        
        # Get labels
        labels = df['generated_sentiment'].values
        
        # Save additional features for analysis
        self.additional_features = df[[
            'user_followers', 'user_friends', 'retweets', 'favorites'
        ]].fillna(0)
        
        return input_ids, attention_masks, labels
    
    def create_dataloaders(self, input_ids, attention_masks, labels, test_size=0.2):
        """Create train and test dataloaders."""
        print("Splitting data into train and test sets...")
        train_inputs, test_inputs, train_masks, test_masks, train_labels, test_labels = train_test_split(
            input_ids, attention_masks, labels, 
            test_size=test_size, 
            random_state=42, 
            stratify=labels
        )
        
        # Create train dataloader
        train_data = TensorDataset(
            torch.tensor(train_inputs),
            torch.tensor(train_masks),
            torch.tensor(train_labels)
        )
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(
            train_data,
            sampler=train_sampler,
            batch_size=self.batch_size
        )
        
        # Create test dataloader
        test_data = TensorDataset(
            torch.tensor(test_inputs),
            torch.tensor(test_masks),
            torch.tensor(test_labels)
        )
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(
            test_data,
            sampler=test_sampler,
            batch_size=self.batch_size
        )
        
        return train_dataloader, test_dataloader
    
    def train(self, train_dataloader, epochs=4, learning_rate=2e-5):
        """Train the model."""
        print("\nInitializing BERT model...")
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3
        ).to(self.device)
        
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        print("\nTraining the model...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for step, batch in enumerate(train_dataloader):
                batch_inputs, batch_masks, batch_labels = batch
                batch_inputs = batch_inputs.to(self.device)
                batch_masks = batch_masks.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                self.model.zero_grad()
                
                outputs = self.model(
                    batch_inputs,
                    attention_mask=batch_masks,
                    labels=batch_labels
                )
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
                if step % 40 == 0:
                    print(f"Epoch {epoch + 1}/{epochs}, Batch {step}, Loss: {loss.item():.4f}")
            
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_train_loss:.4f}")
    
    def evaluate(self, test_dataloader):
        """Evaluate the model's performance."""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")
        
        print("\nEvaluating the model...")
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                batch_inputs, batch_masks, batch_labels = batch
                batch_inputs = batch_inputs.to(self.device)
                batch_masks = batch_masks.to(self.device)
                
                outputs = self.model(
                    batch_inputs,
                    attention_mask=batch_masks
                )
                logits = outputs.logits
                
                predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                true_labels.extend(batch_labels.numpy())
        
        accuracy = accuracy_score(true_labels, predictions)
        print(f"\nAccuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(
            true_labels,
            predictions,
            target_names=['Negative', 'Neutral', 'Positive']
        ))
        
        return accuracy, predictions
    
    def save_model(self, path):
        """Save the trained model and tokenizer."""
        if self.model is None:
            raise ValueError("No model to save!")
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model and tokenizer saved to {path}")
    
    def predict(self, text):
        """Predict sentiment for a single piece of text."""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet!")
        
        cleaned_text = self.clean_tweet(text)
        encoded = self.tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        self.model.eval()
        with torch.no_grad():
            inputs = encoded['input_ids'].to(self.device)
            masks = encoded['attention_mask'].to(self.device)
            outputs = self.model(inputs, attention_mask=masks)
            prediction = torch.argmax(outputs.logits, dim=-1)
        
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        return sentiment_map[prediction.item()]


def main():
    print("Initializing Vaccination Sentiment Analyzer...")
    analyzer = VaccinationSentimentAnalyzer(max_len=128, batch_size=32)
    
    try:
        # Prepare data - Note the change here to catch all three return values
        print("Preparing data...")
        input_ids, attention_masks, labels = analyzer.prepare_data('vaccination_tweets.csv')
        
        print("Creating dataloaders...")
        train_dataloader, test_dataloader = analyzer.create_dataloaders(
            input_ids, attention_masks, labels
        )
        
        print("Starting training...")
        analyzer.train(train_dataloader, epochs=4)
        
        print("Evaluating model...")
        accuracy, predictions = analyzer.evaluate(test_dataloader)
        
        print("Saving model...")
        analyzer.save_model('./saved_model')
        
        # Example predictions
        print("\nTesting with sample tweets...")
        sample_tweets = [
            "The vaccine rollout has been very effective in protecting our community!",
            "Concerned about the side effects of the new vaccine",
            "Just got my vaccine dose today at the local clinic"
        ]
        
        for tweet in sample_tweets:
            sentiment = analyzer.predict(tweet)
            print(f"\nTweet: {tweet}")
            print(f"Sentiment: {sentiment}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please make sure your CSV file is in the correct location and format.")
        raise

if __name__ == "__main__":
    main()