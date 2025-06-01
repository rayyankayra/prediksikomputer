
# Mengimpor library NumPy untuk operasi numerik dan array
import numpy as np
# Mengimpor PyTorch yang merupakan framework untuk deep learning
import torch
# Mengimpor modul Neural Network dari PyTorch
import torch.nn as nn
# Mengimpor Dataset dan DataLoader untuk memproses data
from torch.utils.data import Dataset, DataLoader
# Mengimpor fungsi train_test_split untuk membagi data menjadi training dan testing
from sklearn.model_selection import train_test_split
# Mengimpor MinMaxScaler untuk normalisasi data
from sklearn.preprocessing import MinMaxScaler
# Mengimpor matplotlib untuk visualisasi data
import matplotlib.pyplot as plt
# Untuk mengimpor foto plot dan lain-lain
import os

# Menetapkan nilai seed untuk memastikan hasil yang konsisten setiap kali kode dijalankan
np.random.seed(42)
torch.manual_seed(42)

# Membuat data sintetis (buatan) untuk simulasi
n_samples = 10000  # Jumlah sampel data yang akan dibuat

# Membuat fitur input:
# Ampere - nilai arus listrik acak antara 0.5 dan 5.0
ampere = np.random.uniform(0.5, 5.0, n_samples)
# Watt - dihitung dari ampere dikalikan tegangan (220) dikali faktor daya acak
watt = ampere * 220 * np.random.uniform(0.9, 1.0, n_samples)
# Volt - nilai tegangan listrik dengan distribusi normal dengan mean 220 dan std 2
volt = np.random.normal(220, 2, n_samples)

# Membuat target output (dengan korelasi realistis):
# Menghitung nilai dasar untuk jumlah komputer yang menjalankan YouTube
base_youtube = 0.8 * ampere + 0.5 * watt/220 - 0.2 * (volt-220)
# Menghitung nilai dasar untuk jumlah komputer dalam kondisi idle
base_idle = 0.3 * ampere + 0.2 * watt/220 - 0.1 * (volt-220)

# Menambahkan variasi acak dan membulatkan ke jumlah komputer (tidak boleh negatif)
youtube_computers = np.maximum(0, np.round(base_youtube + np.random.normal(0, 0.5, n_samples)))
idle_computers = np.maximum(0, np.round(base_idle + np.random.normal(0, 0.3, n_samples)))

# Menyusun data:
# X adalah fitur input (ampere, watt, volt)
X = np.column_stack([ampere, watt, volt])
# y adalah target output (jumlah komputer YouTube dan idle)
y = np.column_stack([youtube_computers, idle_computers])

# Normalisasi data ke rentang 0-1 untuk meningkatkan kinerja model
scaler_X = MinMaxScaler()  # Scaler untuk fitur input
scaler_y = MinMaxScaler()  # Scaler untuk target output
X_scaled = scaler_X.fit_transform(X)  # Melakukan fit dan transform fitur
y_scaled = scaler_y.fit_transform(y)  # Melakukan fit dan transform target

# Membuat kelas Dataset kustom sesuai standar PyTorch
class ComputerUsageDataset(Dataset):
    def __init__(self, X, y):
        # Mengkonversi NumPy array ke PyTorch tensor
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        # Mengembalikan jumlah sampel dalam dataset
        return len(self.X)
    
    def __getitem__(self, idx):
        # Mengembalikan pasangan fitur dan target untuk indeks tertentu
        return self.X[idx], self.y[idx]

# Membagi data menjadi data training (80%) dan testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Membuat dataset dan dataloader untuk training dan testing
train_dataset = ComputerUsageDataset(X_train, y_train)
test_dataset = ComputerUsageDataset(X_test, y_test)
# DataLoader dengan batch size 32 dan shuffling untuk data training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# DataLoader untuk data testing (tanpa shuffling)
test_loader = DataLoader(test_dataset, batch_size=32)

# Mendefinisikan arsitektur model-model neural network:

# Model Neural Network Sederhana
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Sequential model dengan 3 layer: input(3) -> hidden(32) -> hidden(16) -> output(2)
        self.model = nn.Sequential(
            nn.Linear(3, 32),  # Layer pertama: 3 input -> 32 neuron
            nn.ReLU(),         # Fungsi aktivasi ReLU
            nn.Linear(32, 16), # Layer kedua: 32 -> 16 neuron
            nn.ReLU(),         # Fungsi aktivasi ReLU
            nn.Linear(16, 2)   # Layer output: 16 -> 2 neuron (jumlah komputer YouTube dan idle)
        )
    
    def forward(self, x):
        # Forward pass: mengembalikan hasil dari model
        return self.model(x)

# Model Neural Network yang Lebih Dalam
class DeepNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Model lebih kompleks dengan layer dan neuron lebih banyak
        self.model = nn.Sequential(
            nn.Linear(3, 128),  # Layer 1: 3 input -> 128 neuron
            nn.ReLU(),          # Fungsi aktivasi ReLU
            nn.Dropout(0.2),    # Dropout 20% untuk mencegah overfitting
            nn.Linear(128, 64), # Layer 2: 128 -> 64 neuron
            nn.ReLU(),          # Fungsi aktivasi ReLU
            nn.Dropout(0.2),    # Dropout 20%
            nn.Linear(64, 32),  # Layer 3: 64 -> 32 neuron
            nn.ReLU(),          # Fungsi aktivasi ReLU
            nn.Linear(32, 2)    # Layer output: 32 -> 2 neuron
        )
    
    def forward(self, x):
        return self.model(x)

# Model LSTM (Long Short-Term Memory) untuk analisis sekuensial
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer LSTM dengan 3 input dan 64 hidden units
        self.lstm = nn.LSTM(input_size=3, hidden_size=64, batch_first=True)
        # Fully connected layer pertama: 64 -> 32 neuron
        self.fc1 = nn.Linear(64, 32)
        # Fungsi aktivasi ReLU
        self.relu = nn.ReLU()
        # Fully connected layer output: 32 -> 2 neuron
        self.fc2 = nn.Linear(32, 2)
    
    def forward(self, x):
        # Menambahkan dimensi sekuensial (batch, sequence, features)
        x = x.unsqueeze(1)  
        # Forward pass melalui LSTM
        lstm_out, _ = self.lstm(x)
        # Mengambil output terakhir dari sequence
        x = self.fc1(lstm_out[:, -1, :])
        # Menerapkan aktivasi ReLU
        x = self.relu(x)
        # Output layer
        x = self.fc2(x)
        return x

# Fungsi untuk melatih model
def train_model(model, train_loader, test_loader, epochs=50):
    # Mendefinisikan loss function (Mean Squared Error)
    criterion = nn.MSELoss()
    # Mendefinisikan optimizer Adam dengan learning rate 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # List untuk menyimpan nilai loss
    train_losses = []
    val_losses = []
    
    # Loop untuk jumlah epoch yang ditentukan
    for epoch in range(epochs):
        # Set model ke mode training
        model.train()
        epoch_loss = 0
        # Iterasi melalui batch data training
        for X_batch, y_batch in train_loader:
            # Reset gradien
            optimizer.zero_grad()
            # Forward pass
            outputs = model(X_batch)
            # Hitung loss
            loss = criterion(outputs, y_batch)
            # Backward pass
            loss.backward()
            # Update parameter
            optimizer.step()
            # Akumulasi loss
            epoch_loss += loss.item()
        
        # Rata-rata loss untuk epoch ini
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validasi model
        model.eval()
        val_loss = 0
        # Disable gradient calculation untuk validasi
        with torch.no_grad():
            # Iterasi melalui batch data testing
            for X_batch, y_batch in test_loader:
                # Forward pass
                outputs = model(X_batch)
                # Hitung dan akumulasi loss validasi
                val_loss += criterion(outputs, y_batch).item()
        # Rata-rata loss validasi
        val_losses.append(val_loss / len(test_loader))
    
    # Kembalikan history loss training dan validasi
    return train_losses, val_losses

# Inisialisasi model-model yang akan dilatih
models = {
    'SimpleNN': SimpleNN(),
    'DeepNN': DeepNN(),
    'LSTM': LSTMModel()
}

# Dictionary untuk menyimpan hasil
results = {}
# Iterasi melalui setiap model
for name, model in models.items():
    print(f"\nTraining {name}...")
    # Melatih model
    train_losses, val_losses = train_model(model, train_loader, test_loader)
    
    # Evaluasi model
    model.eval()
    y_pred = []
    y_true = []
    # Melakukan prediksi pada test set
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            y_pred.extend(outputs.numpy())
            y_true.extend(y_batch.numpy())
    
    # Konversi list ke numpy array
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    
    # Mengembalikan nilai ke skala asli (sebelum normalisasi)
    y_pred_original = scaler_y.inverse_transform(y_pred)
    y_true_original = scaler_y.inverse_transform(y_true)
    
    # Menghitung metrik evaluasi:
    # MAE (Mean Absolute Error) untuk YouTube dan idle
    mae_youtube = np.mean(np.abs(y_pred_original[:, 0] - y_true_original[:, 0]))
    mae_idle = np.mean(np.abs(y_pred_original[:, 1] - y_true_original[:, 1]))
    # Standar deviasi error untuk YouTube dan idle
    std_youtube = np.std(np.abs(y_pred_original[:, 0] - y_true_original[:, 0]))
    std_idle = np.std(np.abs(y_pred_original[:, 1] - y_true_original[:, 1]))
    
    # R² (koefisien determinasi) untuk YouTube dan idle
    r2_youtube = 1 - np.sum((y_true_original[:, 0] - y_pred_original[:, 0])**2) / np.sum((y_true_original[:, 0] - np.mean(y_true_original[:, 0]))**2)
    r2_idle = 1 - np.sum((y_true_original[:, 1] - y_pred_original[:, 1])**2) / np.sum((y_true_original[:, 1] - np.mean(y_true_original[:, 1]))**2)
    
    # Menyimpan hasil evaluasi
    results[name] = {
        'mae_youtube': mae_youtube,
        'mae_idle': mae_idle,
        'std_youtube': std_youtube,
        'std_idle': std_idle,
        'r2_youtube': r2_youtube,
        'r2_idle': r2_idle,
        'train_losses': train_losses,
        'val_losses': val_losses
    }

# Membuat visualisasi hasil
plt.figure(figsize=(15, 5))

# Plot Training Loss
plt.subplot(1, 2, 1)  # 1 baris, 2 kolom, plot pertama
# Plot loss training dan validasi untuk setiap model
for name, result in results.items():
    plt.plot(result['train_losses'], label=f'{name} Training')
    plt.plot(result['val_losses'], label=f'{name} Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Plot Perbandingan Performa Prediksi 
plt.subplot(1, 2, 2)  # Plot kedua
x = np.arange(3)  # Posisi x untuk bar plot
width = 0.25  # Lebar bar

# Nilai MAE untuk YouTube dan idle dari setiap model
youtube_mae = [results[model]['mae_youtube'] for model in ['SimpleNN', 'DeepNN', 'LSTM']]
idle_mae = [results[model]['mae_idle'] for model in ['SimpleNN', 'DeepNN', 'LSTM']]

# Membuat bar plot
plt.bar(x - width/2, youtube_mae, width, label='YouTube MAE')
plt.bar(x + width/2, idle_mae, width, label='Idle MAE')
plt.xlabel('Model')
plt.ylabel('Mean Absolute Error')
plt.title('Prediction Performance Comparison')
plt.xticks(x, ['SimpleNN', 'DeepNN', 'LSTM'])
plt.legend()
plt.grid(True)

# Mengatur layout dan menyimpan grafik
print("📌 Mulai proses penyimpanan plot...")
plt.tight_layout()
save_path = os.path.abspath("simulation_results.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ Gambar disimpan di: {save_path}")
plt.close()

# Mencetak hasil detail untuk setiap model
print("\nDetailed Results:")
print("-" * 50)
for name, result in results.items():
    print(f"\n{name}:")
    print(f"YouTube MAE: {result['mae_youtube']:.2f} ± {result['std_youtube']:.2f}")
    print(f"Idle MAE: {result['mae_idle']:.2f} ± {result['std_idle']:.2f}")
    print(f"YouTube R²: {result['r2_youtube']:.4f}")
    print(f"Idle R²: {result['r2_idle']:.4f}")

# Fungsi untuk analisis distribusi error
def analyze_error_distribution(model, test_loader, scaler_X, scaler_y):
    # Mode evaluasi
    model.eval()
    predictions = []
    targets = []
    inputs = []
    
    # Mengumpulkan prediksi, target, dan input
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions.extend(outputs.numpy())
            targets.extend(y_batch.numpy())
            inputs.extend(X_batch.numpy())
    
    # Konversi ke numpy array
    predictions = np.array(predictions)
    targets = np.array(targets)
    inputs = np.array(inputs)