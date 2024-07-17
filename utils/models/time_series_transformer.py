import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

import json
import joblib

from bayes_opt import BayesianOptimization
import shap

import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ConvLayer(nn.Module):
    def __init__(self, input_dim):
        super(ConvLayer, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x.permute(0, 2, 1)  # (batch, seq_len, input_dim)

class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super(TemporalAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attention = self.softmax(scores)
        
        return torch.matmul(attention, v)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=512, nhead=16, num_layers=6, dropout=0.1, output_dim=7):
        super(TimeSeriesTransformer, self).__init__()
        self.conv_layer = ConvLayer(input_dim)
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.temporal_attention = TemporalAttention(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, output_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask=None):
        src = self.conv_layer(src)
        src = self.embedding(src)
        src = self.pos_encoder(src)
        src = self.temporal_attention(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output[:, -1, :])
        return output

def create_attention_mask(sequence_length):
    mask = torch.triu(torch.ones(sequence_length, sequence_length), diagonal=1).bool()
    return mask

def prepare_data(file_path, sequence_length=30, prediction_window=7, step=1):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values('Date')

    features = ['Open', 'High', 'Low', 'Close_lag_7', 'Close_lag_14',
                'Close_rolling_7', 'Close_rolling_14', 'MA(7)', 'MA(25)', 'MA(99)',
                'RSI', 'Vol(USDT)', 'Ensemble_Sentiment', 'Sentiment_7day_MA',
                'Sentiment_Change', 'Sentiment_Volatility']
    target = 'Close'

    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    df[features] = scaler_features.fit_transform(df[features])
    df[target] = scaler_target.fit_transform(df[[target]])

    X, y = [], []
    for i in range(0, len(df) - sequence_length - prediction_window + 1, step):
        X.append(df[features].iloc[i:i+sequence_length].values)
        y.append(df[target].iloc[i+sequence_length:i+sequence_length+prediction_window].values)

    X = np.array(X)
    y = np.array(y)

    return torch.FloatTensor(X), torch.FloatTensor(y), scaler_features, scaler_target

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-8))

def evaluate_model(model, data_loader, scaler_target, criterion):
    model.eval()
    total_loss = 0
    true_values = []
    predictions = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            mask = create_attention_mask(batch_X.size(1)).to(batch_X.device)
            outputs = model(batch_X, mask)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
            true_values.append(batch_y.numpy())
            predictions.append(outputs.numpy())
    
    true_values = np.concatenate(true_values)
    predictions = np.concatenate(predictions)
    
    true_values = scaler_target.inverse_transform(true_values)
    predictions = scaler_target.inverse_transform(predictions)
    
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mean_squared_error(true_values, predictions))
    smape_value = smape(true_values, predictions)
    
    return total_loss / len(data_loader), mae, rmse, smape_value

def train_model(model, train_loader, val_loader, scaler_target, epochs=100, lr=0.0001, patience=10):
    criterion = nn.HuberLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            mask = create_attention_mask(batch_X.size(1)).to(batch_X.device)
            outputs = model(batch_X, mask)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        val_loss, val_mae, val_rmse, val_smape = evaluate_model(model, val_loader, scaler_target, criterion)

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}, Val SMAPE: {val_smape:.2f}%')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

def k_fold_cross_validation(X, y, model_class, n_splits=5, epochs=100, lr=0.0001, patience=10, batch_size=64):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []

    for fold, (train_index, val_index) in enumerate(tscv.split(X), 1):
        print(f"Fold {fold}")

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        input_dim = X_train.shape[2]
        model = model_class(input_dim=input_dim, d_model=512, nhead=16, num_layers=6, output_dim=prediction_window)

        train_model(model, train_loader, val_loader, scaler_target, epochs=epochs, lr=lr, patience=patience)

        _, mae, rmse, smape_value = evaluate_model(model, val_loader, scaler_target, nn.HuberLoss())
        fold_metrics.append((mae, rmse, smape_value))

        print(f"Fold {fold} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, SMAPE: {smape_value:.2f}%")

    avg_mae = np.mean([m[0] for m in fold_metrics])
    avg_rmse = np.mean([m[1] for m in fold_metrics])
    avg_smape = np.mean([m[2] for m in fold_metrics])

    print(f"\nAverage Metrics:")
    print(f"MAE: {avg_mae:.4f}")
    print(f"RMSE: {avg_rmse:.4f}")
    print(f"SMAPE: {avg_smape:.2f}%")

    return fold_metrics


def model_objective(d_model, nhead, num_layers, dropout, learning_rate):
    # Convert float parameters to int
    d_model = int(d_model)
    nhead = int(nhead)
    num_layers = int(num_layers)
    
    # Ensure nhead is even
    nhead = nhead if nhead % 2 == 0 else nhead + 1
    
    # Ensure d_model is even and divisible by nhead
    d_model = ((d_model + 1) // 2) * 2  # Make d_model even
    d_model = (d_model // nhead) * nhead
    
    model = TimeSeriesTransformer(input_dim=X.shape[2], d_model=d_model, nhead=nhead, 
                                  num_layers=num_layers, dropout=dropout, output_dim=prediction_window)
    
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    train_model(model, train_loader, val_loader, scaler_target, epochs=50, lr=learning_rate, patience=10)
    
    _, _, _, smape_value = evaluate_model(model, val_loader, scaler_target, nn.HuberLoss())
    
    return -smape_value  # Return negative SMAPE as we want to maximize the objective

def optimize_hyperparameters():
    pbounds = {
        'd_model': (64, 512),
        'nhead': (2, 16),
        'num_layers': (1, 8),
        'dropout': (0.1, 0.5),
        'learning_rate': (1e-4, 1e-2)
    }

    optimizer = BayesianOptimization(
        f=model_objective,
        pbounds=pbounds,
        random_state=1,
    )

    optimizer.maximize(
        init_points=5,
        n_iter=25,
    )

    return optimizer.max


def interpret_model(model, X, y, features):
    model.eval()
    
    # Convert tensors to numpy arrays
    X_np = X.numpy()
    y_np = y.numpy()
    
    # Calculate baseline performance
    with torch.no_grad():
        baseline_preds = model(X).numpy()
    baseline_mae = mean_absolute_error(y_np, baseline_preds)
    
    # Calculate feature importance
    feature_importance = []
    for i in range(X.shape[2]):  # Iterate over features
        X_permuted = X_np.copy()
        X_permuted[:, :, i] = np.random.permutation(X_permuted[:, :, i])
        
        with torch.no_grad():
            permuted_preds = model(torch.FloatTensor(X_permuted)).numpy()
        permuted_mae = mean_absolute_error(y_np, permuted_preds)
        
        importance = permuted_mae - baseline_mae
        feature_importance.append(importance)
    
    # Normalize feature importance
    feature_importance = np.array(feature_importance)
    feature_importance = feature_importance / np.sum(np.abs(feature_importance))
    
    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_importance = feature_importance[sorted_idx]
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_features)), sorted_importance)
    plt.yticks(range(len(sorted_features)), sorted_features)
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    # Print feature importance
    for feature, importance in zip(sorted_features, sorted_importance):
        print(f"{feature}: {importance:.4f}")