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
st.title("🔌 Prediksi Jumlah Komputer Berdasarkan Data Listrik (LSTM)")

# Set seed
np.random.seed(42)
torch.manual_seed(42)

# Generate data sintetis
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

# Definisi model LSTM
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)
    def forward(self, x):
        x = x.unsqueeze(1)  # batch, seq_len=1, features
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out[:, -1, :])
        x = self.relu(x)
        return self.fc2(x)

@st.cache_data(show_spinner=False)
def train_lstm_model():
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 50
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
            val_loss += criterion(ourputs, y_batch).item()
    val_losses.append(val_loss / len(test_loader))

# Sidebar Train dari user
st.sidebar.header("Training Model LSTM")
train_button = st.sidebar.button("Train Model LSTM")

if train_button:
    with st.spinner("Melatih model LSTM..."):
        model, train_losses, val_losses = train_lstm_model()
    st.success("Training selesai!")

    # Tampilkan grafik loss asli
    fig, ax = plt.subplots()
    ax.plot(train_losses, label='Train Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Grafik Loss Training & Validasi Model LSTM')
    ax.legend()
    st.pyplot(fig)

    # Simpan model ke variabel global supaya bisa dipakai prediksi input user
    st.session_state['model'] = model

# Sidebar input dari user
st.sidebar.header("Input Data Listrik")
input_ampere = st.sidebar.number_input("Ampere (A)", min_value=0.5, max_value=5.0, value=1.0, step=0.1)
input_watt = st.sidebar.number_input("Watt (W)", min_value=50.0, max_value=1200.0, value=220.0, step=10.0)
input_volt = st.sidebar.number_input("Volt (V)", min_value=210.0, max_value=230.0, value=220.0, step=0.5)

predict_button = st.sidebar.button("Prediksi Jumlah Komputer")

if predict_button:
    if 'model' not in st.session_state:
        st.warning("Silahkan latih model dulu dengan klik tombol 'Train Model LSTM'.")
    else:
        model = st.session_state['model']
        model.eval()
        
        # Buat input array dan scaling
        input_arr = np.array([[input_ampere, input_watt, input_volt]])
        input_scaled = scaler_X.transform(input_arr)
        input_tensor = torch.FloatTensor(input_scaled)

        # Prediksi dengan model
        with torch.no_grad():
            pred_scaled = model(input_tensor).numpy()
    
        # Inverse scaling ke nilai asli
        pred_original = scaler_y.inverse_transform(pred_scaled)
        pred_youtube = max(0, pred_original[0][0])
        pred_idle = max(0, pred_original[0][1])
    
        st.subheader("Hasil Prediksi")
        st.write(f"Jumlah komputer aktif YouTube diperkirakan: **{pred_youtube:.1f}**")
        st.write(f"Jumlah komputer idle diperkirakan: **{pred_idle:.1f}**")

# Tampilkan contoh plot loss training (opsional, hanya visualisasi)
st.write("---")
st.subheader("Contoh Grafik Loss Training Model (Sample)")
fig, ax = plt.subplots()
ax.plot(np.linspace(1,50,50), np.linspace(0.1,0.01,50), label="Train Loss (Contoh)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
st.pyplot(fig)
