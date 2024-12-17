import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

class LSTMTextClassifier:
    def __init__(self, embedding_dim=50, hidden_dim=100, num_layers=1, dropout=0.3):
        """
        Initialize LSTM Text Classifier
        
        Parameters:
        - embedding_dim: Dimension of the embedding layer
        - hidden_dim: Number of features in the hidden state of LSTM
        - num_layers: Number of LSTM layers
        - dropout: Dropout rate for regularization
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vectorizer = CountVectorizer(max_features=1000)
        self.encoder = LabelEncoder()
        
        # LSTM specific parameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.model = None

    def prepare_data(self, texts, labels):
        """
        Prepare and vectorize input texts and labels
        
        Parameters:
        - texts: List of input text documents
        - labels: Corresponding labels
        
        Returns:
        - X: Vectorized text features as PyTorch tensor
        - y: Encoded labels as PyTorch tensor
        """
        # Vectorisation des textes
        X = self.vectorizer.fit_transform(texts).toarray()
        y = self.encoder.fit_transform(labels)
        
        # Conversion en tensors
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        return X, y

    def build_model(self, input_dim):
        """
        Build LSTM neural network model
        
        Parameters:
        - input_dim: Dimension of input features
        
        Returns:
        - Initialized LSTM model
        """
        class LSTMNet(nn.Module):
            def __init__(self, input_dim, embedding_dim, hidden_dim, 
                         num_layers, num_classes, dropout):
                super().__init__()
                
                # Embedding layer
                self.embedding = nn.Linear(input_dim, embedding_dim)
                
                # LSTM layers
                self.lstm = nn.LSTM(
                    input_size=embedding_dim, 
                    hidden_size=hidden_dim, 
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout if num_layers > 1 else 0
                )
                
                # Fully connected output layer
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, num_classes)
                )
            
            def forward(self, x):
                # Embedding
                x = self.embedding(x)
                
                # Add sequence dimension for LSTM
                x = x.unsqueeze(1)
                
                # LSTM processing
                lstm_out, _ = self.lstm(x)
                
                # Take the last time step
                x = lstm_out[:, -1, :]
                
                # Output layer
                return self.fc(x)

        return LSTMNet(
            input_dim=input_dim, 
            embedding_dim=self.embedding_dim, 
            hidden_dim=self.hidden_dim, 
            num_layers=self.num_layers,
            num_classes=len(self.encoder.classes_),
            dropout=self.dropout
        ).to(self.device)

    def train(self, X, y, epochs=50, lr=0.001, batch_size=32):
        """
        Train the LSTM model
        
        Parameters:
        - X: Input features
        - y: Labels
        - epochs: Number of training epochs
        - lr: Learning rate
        - batch_size: Size of training batches
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build model
        self.model = self.build_model(X.shape[1])
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
            
            # Validation
            self.model.eval()
            total_val_loss = 0
            correct_predictions = 0
            total_predictions = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    total_val_loss += loss.item()
                    
                    # Accuracy calculation
                    _, predicted = torch.max(outputs, 1)
                    total_predictions += batch_y.size(0)
                    correct_predictions += (predicted == batch_y).sum().item()
            
            # Print epoch statistics
            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            val_accuracy = correct_predictions / total_predictions
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}:')
                print(f'  Train Loss: {avg_train_loss:.4f}')
                print(f'  Val Loss: {avg_val_loss:.4f}')
                print(f'  Val Accuracy: {val_accuracy:.4f}')

    def predict(self, texts):
        """
        Make predictions for new texts
        
        Parameters:
        - texts: List of input text documents
        
        Returns:
        - Predicted labels
        """
        # Vectoriser les nouveaux textes
        X = self.vectorizer.transform(texts).toarray()
        X = torch.FloatTensor(X).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        
        # Convertir en labels originaux
        return self.encoder.inverse_transform(predicted.cpu().numpy())