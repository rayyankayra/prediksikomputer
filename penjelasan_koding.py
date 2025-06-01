import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import streamlit as st

# Konfigurasi tampilan
st.set_page_config(layout="wide")
st.title("🔌 Prediksi Jumlah Komputer Berdasarkan Data Listrik")

# Set seed
np.random.seed(42)
torch.manual_seed(42)

# Generate data
n_samples = 10000
ampere = np.random.uniform(0.5, 5.0, n_samples)
watt = ampere * 220 * np.random.uniform(0.9, 1.0, n_samples)
volt = np.random.normal(220, 2, n_samples)

base_youtube = 0.8 * ampere + 0.5 * watt/220 - 0.2 * (volt-220)
base_idle = 0.3 * ampere + 0.2 * watt/220 - 0.1 * (volt-220)

youtube_computers = np.maximum(0, np.round(base_youtube + np.random.normal(0, 0.5, n_samples)))
idle_computers = np.maximum(0, np.round(base_idle + np.random.normal(0, 0.3, n_samples)))

X = np.column_stack([ampere, watt, volt])
y = np.column_stack([youtube_computers, idle_computers])

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

class ComputerUsageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
train_dataset = ComputerUsageDataset(X_train, y_train)
test_dataset = ComputerUsageDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Model definitions
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        return self.model(x)

class DeepNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.model(x)

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out[:, -1, :])
        x = self.relu(x)
        return self.fc2(x)

def train_model(model, train_loader, test_loader, epochs=50):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
        val_losses.append(val_loss / len(test_loader))
    return train_losses, val_losses

# Sidebar: Pilih model
model_choice = st.sidebar.selectbox("Pilih Model", ["SimpleNN", "DeepNN", "LSTM"])
train_button = st.sidebar.button("Train Model")

if train_button:
    model_map = {
        "SimpleNN": SimpleNN(),
        "DeepNN": DeepNN(),
        "LSTM": LSTMModel()
    }
    model = model_map[model_choice]
    with st.spinner("Melatih model..."):
        train_losses, val_losses = train_model(model, train_loader, test_loader)

        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                y_pred.extend(outputs.numpy())
                y_true.extend(y_batch.numpy())

        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        y_pred_original = scaler_y.inverse_transform(y_pred)
        y_true_original = scaler_y.inverse_transform(y_true)

        mae_youtube = np.mean(np.abs(y_pred_original[:, 0] - y_true_original[:, 0]))
        mae_idle = np.mean(np.abs(y_pred_original[:, 1] - y_true_original[:, 1]))
        r2_youtube = 1 - np.sum((y_true_original[:, 0] - y_pred_original[:, 0])**2) / np.sum((y_true_original[:, 0] - np.mean(y_true_original[:, 0]))**2)
        r2_idle = 1 - np.sum((y_true_original[:, 1] - y_pred_original[:, 1])**2) / np.sum((y_true_original[:, 1] - np.mean(y_true_original[:, 1]))**2)

    # Plotting loss
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_title("Loss Training vs Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    st.pyplot(fig)

    # Menampilkan hasil
    st.subheader(f"📊 Hasil Evaluasi Model: {model_choice}")
    st.metric("YouTube MAE", f"{mae_youtube:.2f}")
    st.metric("Idle MAE", f"{mae_idle:.2f}")
    st.write(f"R² YouTube: {r2_youtube:.4f}")
    st.write(f"R² Idle: {r2_idle:.4f}")
