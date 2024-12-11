import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def preprocess_data(df):
    """Preprocess the dataframe"""
    df['Left_Lane_Vehicles'] = df['Left_Lane_Vehicles'].fillna('').apply(lambda x: [] if x == '' else x.strip('[]').split(','))
    df['Right_Lane_Vehicles'] = df['Right_Lane_Vehicles'].fillna('').apply(lambda x: [] if x == '' else x.strip('[]').split(','))
    df['Time_To_Collision'] = df['Time_To_Collision'].replace([np.inf, -np.inf], 1e6)
    return df

class MaxEntIRLModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Process each timestep through feature network
        x = x.view(-1, self.input_dim)  # Combine batch and sequence dimensions
        x = self.feature_net(x)
        x = x.view(batch_size, seq_len, self.hidden_dim)  # Restore batch and sequence dimensions
        
        # Process sequence through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use last LSTM output
        last_output = lstm_out[:, -1]
        
        # Final prediction
        out = self.fc(last_output)
        return out

class TrajectoryDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Create feature vector [x, y, vx, vy, ax]
        features = torch.tensor([
            row['Position_X'],
            row['Position_Y'],
            row['Velocity_X'],
            row['Velocity_Y'],
            row['Acceleration_X']
        ], dtype=torch.float32)
        
        # Create sequence by repeating and adding some variation
        sequence = features.unsqueeze(0).repeat(30, 1)  # 30 timesteps
        
        # Add time-based variations to position and velocity
        time_steps = torch.arange(30, dtype=torch.float32) * 0.2
        sequence[:, 0] += sequence[:, 2] * time_steps  # x position changes with velocity
        sequence[:, 1] += sequence[:, 3] * time_steps  # y position changes with velocity
        
        # Target is 1 for expert trajectories (assuming all data is from expert)
        target = torch.tensor(1.0, dtype=torch.float32)
        
        return sequence, target

def train(model, train_loader, num_epochs=10, learning_rate=0.001):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, targets in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs.squeeze(), targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
    return model

def main():
    print("Loading data...")
    data = pd.read_csv('traffic_data.csv')
    data = preprocess_data(data)
    
    print("Creating dataset...")
    dataset = TrajectoryDataset(data)
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    print("Initializing MaxEnt IRL model...")
    model = MaxEntIRLModel().to(device)
    
    print("Starting training...")
    trained_model = train(model, train_loader)
    
    print("Saving model...")
    torch.save(trained_model.state_dict(), 'irl_model.pth')
    print("Training complete!")

if __name__ == "__main__":
    main()
