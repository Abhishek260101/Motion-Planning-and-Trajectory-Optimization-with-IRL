import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'  # Fix OpenMP error

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import random

class MaxEntIRLNet(nn.Module):
    def __init__(self, feature_dims):
        super().__init__()

        # Add new feature processors
        self.lane_change_process = self._make_feature_processor(feature_dims['lane_change'])
        self.overtaking_process = self._make_feature_processor(feature_dims['overtaking'])
                
        # Feature processors with correct dimensions
        self.ttc_process = self._make_feature_processor(feature_dims['ttc'])
        self.acc_process = self._make_feature_processor(feature_dims['acc_info'])
        self.dynamics_process = self._make_feature_processor(feature_dims['dynamics'])
        self.surroundings_process = self._make_feature_processor(feature_dims['surroundings'])
        self.lane_change_process = self._make_feature_processor(feature_dims['lane_change'])
        self.overtaking_process = self._make_feature_processor(feature_dims['overtaking'])
        
        # Fixed hidden size
        hidden_size = 64  # Changed from previous value
        
        # LSTM layers with consistent dimensions
        self.ttc_lstm = nn.LSTM(feature_dims['ttc'], hidden_size, batch_first=True)
        self.acc_lstm = nn.LSTM(feature_dims['acc_info'], hidden_size, batch_first=True)
        self.dynamics_lstm = nn.LSTM(feature_dims['dynamics'], hidden_size, batch_first=True)
        self.surroundings_lstm = nn.LSTM(feature_dims['surroundings'], hidden_size, batch_first=True)
        self.lane_change_lstm = nn.LSTM(feature_dims['lane_change'], hidden_size, batch_first=True)
        self.overtaking_lstm = nn.LSTM(feature_dims['overtaking'], hidden_size, batch_first=True)
        
        # Total hidden size for attention
        total_hidden = hidden_size * 6
        self.attention = nn.MultiheadAttention(embed_dim=total_hidden, num_heads=8)
   
        self.batch_norm = nn.BatchNorm1d(sum(feature_dims.values()))
        self.dropout = nn.Dropout(0.3)  # Increase dropout

        # Final scoring network
        self.score_net = nn.Sequential(
            nn.BatchNorm1d(total_hidden),
            nn.Linear(total_hidden, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        
        # Feature importance weights (learned)
        self.feature_weights = nn.Parameter(torch.ones(6))
        
    def _make_feature_processor(self, input_dim):
        return nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.BatchNorm1d(input_dim),
            nn.Dropout(0.3)
        )

    
    def forward(self, features, mask=None):
        batch_size = features['ttc'].shape[0]
        
        # Process features
        ttc = self.ttc_process(features['ttc'])
        acc = self.acc_process(features['acc_info'])
        dynamics = self.dynamics_process(features['dynamics'])
        surroundings = self.surroundings_process(features['surroundings'])
        lane_change = self.lane_change_process(features['lane_change'])
        overtaking = self.overtaking_process(features['overtaking'])
        
        # Process through LSTMs
        ttc_out, _ = self.ttc_lstm(ttc.unsqueeze(1))
        acc_out, _ = self.acc_lstm(acc.unsqueeze(1))
        dynamics_out, _ = self.dynamics_lstm(dynamics.unsqueeze(1))
        surroundings_out, _ = self.surroundings_lstm(surroundings.unsqueeze(1))
        lane_change_out, _ = self.lane_change_lstm(lane_change.unsqueeze(1))
        overtaking_out, _ = self.overtaking_lstm(overtaking.unsqueeze(1))
        
        # Combine feature representations
        combined = torch.cat([
            ttc_out[:, -1],
            acc_out[:, -1],
            dynamics_out[:, -1],
            surroundings_out[:, -1],
            lane_change_out[:, -1],
            overtaking_out[:, -1]
        ], dim=1)
        
        # Self-attention
        combined = combined.unsqueeze(0)
        if mask is not None:
            attended, _ = self.attention(combined, combined, combined, key_padding_mask=mask)
        else:
            attended, _ = self.attention(combined, combined, combined)
        combined = attended.squeeze(0)
        
        # Generate rewards
        rewards = self.score_net(combined)
        
        return rewards

class TrajectoryDataset(Dataset):
    def __init__(self, data_path, num_trajectories=10):
        self.data = pd.read_csv(data_path)
        self.num_trajectories = num_trajectories
        self.vehicle_groups = self._group_vehicle_data()
        self.features = self._extract_features()
        self.expert_trajectories = self._identify_expert_trajectories()
        
    def _group_vehicle_data(self):
        """Group data by vehicle ID to track trajectories"""
        return {vid: group for vid, group in self.data.groupby('Vehicle_ID')}
    
    def _extract_features(self):
        """Extract and normalize features with proper inf/boolean handling"""
        features = {}
        
        # Replace inf values with large numbers for numerical stability
        data = self.data.replace([np.inf, -np.inf], np.nan)
        
        # Convert boolean columns to float
        bool_columns = ['Current_Lane_Change_State', 'Lane_Change_Success', 'Collision_Occurred']
        for col in bool_columns:
            if col in data.columns:
                data[col] = data[col].astype(float)
        
        # TTC features
        ttc_cols = ['Front_Vehicle_Distance', 'Front_Vehicle_Speed']
        features['ttc'] = self._normalize_feature_group(data[ttc_cols], fill_value=100.0)
        
        # ACC features
        acc_cols = ['Current_Speed', 'Front_Vehicle_Distance', 'Front_Vehicle_Speed', 'Time_To_Collision_Front']
        features['acc_info'] = self._normalize_feature_group(data[acc_cols], fill_value=100.0)
        
        # Vehicle dynamics
        dynamics_cols = ['Current_Speed', 'Current_Acceleration', 'Current_Lane']
        features['dynamics'] = self._normalize_feature_group(data[dynamics_cols], fill_value=0.0)
        
        # Surrounding vehicles
        surrounding_cols = [
            'Left_Front_Vehicle_Distance', 'Left_Front_Vehicle_Speed',
            'Left_Rear_Vehicle_Distance', 'Left_Rear_Vehicle_Speed',
            'Right_Front_Vehicle_Distance', 'Right_Front_Vehicle_Speed',
            'Right_Rear_Vehicle_Distance', 'Right_Rear_Vehicle_Speed'
        ]
        features['surroundings'] = self._normalize_feature_group(data[surrounding_cols], fill_value=100.0)

        # Add lane-change specific features
        features['lane_change'] = self._normalize_feature_group(data[[
            'Time_Since_Last_Change',
            'Current_Lane_Change_State',
            'Lane_Change_Success',
            'Current_Lane'
        ]], fill_value=0.0)
        
        # Add overtaking-specific features
        features['overtaking'] = self._normalize_feature_group(data[[
            'Front_Vehicle_Speed',
            'Current_Speed',
            'Time_Following_Slower_Vehicle',
            'Successful_Overtakes'
        ]], fill_value=0.0)

        return features
    
    def _normalize_feature_group(self, data, fill_value=0.0):
        """Normalize features with proper handling of inf/nan values and boolean data"""
        normalized = data.copy()
        
        # Fill inf/nan values
        normalized = normalized.fillna(fill_value)
        
        # Normalize each column independently
        for col in normalized.columns:
            col_data = normalized[col].values
            
            # Convert boolean columns to int
            if col_data.dtype == bool:
                col_data = col_data.astype(float)
            
            # Only normalize if there's variation and not boolean
            if np.std(col_data) > 0:
                try:
                    # Use robust scaling
                    q75, q25 = np.percentile(col_data, [75, 25])
                    iqr = q75 - q25 + 1e-8  # Add small epsilon to avoid division by zero
                    normalized[col] = (col_data - q25) / iqr
                except TypeError:
                    # If normalization fails, just convert to float
                    normalized[col] = col_data.astype(float)
        
        return normalized.values
    
    def _identify_expert_trajectories(self):
        """Identify expert trajectories using multiple metrics"""
        expert_scores = np.zeros(len(self.data))
        
        for vid, group in self.vehicle_groups.items():
            # Calculate trajectory-level metrics
            avg_speed = group['Current_Speed'].mean()
            collision_free = not group['Collision_Occurred'].any()
            safe_distances = group['Safety_Margin'].mean()
            successful_changes = group['Lane_Change_Success'].all()

            lane_changes = group['Current_Lane'].diff().abs().sum() > 0
            successful_overtakes = group['Successful_Overtakes'].sum() > 0

            # Combine metrics into an expert score
            score = (
                (avg_speed > 0) * 0.2 +                  # Moving vehicles
                collision_free * 0.3 +                    # No collisions
                (safe_distances > 5.0) * 0.2 +           # Safe distance
                successful_changes * 0.15 +              # Lane changes
                lane_changes * 0.1 +                     # Attempted lane changes
                successful_overtakes * 0.05              # Successful overtakes
            )
            
            # Assign scores to all timestamps of this vehicle
            mask = self.data['Vehicle_ID'] == vid
            expert_scores[mask] = score
        
        # Convert to tensor
        return torch.FloatTensor(expert_scores)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = {k: torch.FloatTensor(v[idx]) for k, v in self.features.items()}
        return features, self.expert_trajectories[idx]

def maxent_irl_loss(rewards, expert_trajectories, features, gamma=2.0, eps=1e-8):
    if rewards.dim() == 1:
        rewards = rewards.unsqueeze(1)
    
    # Clip rewards for stability
    rewards = torch.clamp(rewards, -10.0, 10.0)
    
    # Normalize rewards with more stable epsilon
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    
    temperature = 2.0
    logits = rewards / temperature
    
    log_probs = torch.log_softmax(logits, dim=0)
    
    expert_mask = expert_trajectories > 0.5
    if not expert_mask.any():
        return torch.tensor(0.0, requires_grad=True, device=rewards.device)
    
    expert_log_probs = log_probs[expert_mask]
    
    probs = torch.exp(log_probs)
    focal_weights = torch.clamp((1 - probs[expert_mask]) ** gamma, 0.0, 1.0)
    
    base_loss = -focal_weights * expert_log_probs
    base_loss = torch.nan_to_num(base_loss, nan=0.0, posinf=1.0, neginf=0.0)
    
    lane_change_reward = 0.5 * torch.mean(torch.abs(features['lane_change']))  # Increased from 0.3
    overtaking_reward = 0.1 * torch.mean(torch.relu(features['overtaking'][:, 1] - features['overtaking'][:, 0]))
    safe_distance_reward = 0.1 * torch.mean(torch.relu(features['ttc'][:, 0] - 10.0))
    
    # Notice the sign change for lane_change_reward (now subtracting like other rewards)
    total_loss = (
        base_loss.mean() - 
        lane_change_reward -  # Changed from + to -
        overtaking_reward - 
        safe_distance_reward
    )
    
    return torch.clamp(total_loss, 0.0, 10.0)

def augment_training_data(features, expert_trajectories):
    """Augment training data with realistic variations"""
    augmented_features = {}
    
    for key, value in features.items():
        # Convert to tensor if not already
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        
        # Add random noise to distances
        if key in ['ttc', 'surroundings']:
            noise = torch.randn_like(value) * 0.1
            value = value + noise
            
        # Add noise to speeds
        if key in ['acc_info', 'dynamics']:
            noise = torch.randn_like(value) * 0.05
            value = value + noise
        
        # Randomly modify lane change features
        if key == 'lane_change' and random.random() < 0.2:
            value = -value  # Flip lane change direction
            
        # Ensure values stay reasonable
        value = torch.clamp(value, -10.0, 10.0)
        augmented_features[key] = value
    
    # Occasionally flip expert trajectory labels for robustness
    if random.random() < 0.05:
        expert_trajectories = 1 - expert_trajectories
    
    return augmented_features, expert_trajectories

class TrainingManager:
    def __init__(self, model, train_loader, val_loader=None, 
                 lr=1e-3, weight_decay=5e-4):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer with weight decay
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,  # More gradual reduction
            patience=5,   # Increased patience
            verbose=True,
            min_lr=1e-6
        )

        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=100
        )

        # Training metrics
        # Early stopping
        self.early_stopping_counter = 0
        self.early_stopping_patience = 10
        self.best_val_loss = float('inf')
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        batch_count = 0
        epoch_losses = []  # To store all losses for this epoch
        
        pbar = tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}')
        
        accumulation_steps = 4
        self.optimizer.zero_grad()
        
        for batch_idx, (features, expert_trajectories) in enumerate(self.train_loader):
            try:
                # Data augmentation
                if random.random() < 0.3:
                    features, expert_trajectories = augment_training_data(features, expert_trajectories)
                
                # Move data to device
                features = {k: v.to(self.device) for k, v in features.items()}
                expert_trajectories = expert_trajectories.to(self.device)
                
                # Forward pass
                rewards = self.model(features)
                loss = maxent_irl_loss(rewards, expert_trajectories, features)
                
                # Handle invalid loss
                if not torch.isfinite(loss):
                    print(f"Warning: Invalid loss detected ({loss.item()}). Skipping batch.")
                    continue
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()
                
                # Record the unscaled loss
                current_loss = loss.item() * accumulation_steps
                epoch_losses.append(current_loss)
                epoch_loss += current_loss
                batch_count += 1
                
                # Update weights after accumulation steps
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({'loss': f'{current_loss:.4f}'})
                
                # Log every 100 batches
                if batch_idx % 100 == 0:
                    self.train_losses.append(np.mean(epoch_losses[-100:]))  # Record mean of last 100 losses
                    
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        # Handle last incomplete accumulation batch
        if (batch_idx + 1) % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        pbar.close()
        
        final_epoch_loss = epoch_loss / (batch_count + 1e-8)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Average Training Loss: {final_epoch_loss:.4f}")
        print(f"Number of batches processed: {batch_count}")
        
        return final_epoch_loss
    
    def validate(self):
        if self.val_loader is None:
            return None
            
        self.model.eval()
        val_loss = 0
        batch_count = 0
        
        with torch.no_grad():
            for features, expert_trajectories in self.val_loader:
                features = {k: v.to(self.device) for k, v in features.items()}
                expert_trajectories = expert_trajectories.to(self.device)
                
                rewards = self.model(features)
                loss = maxent_irl_loss(rewards, expert_trajectories, features)
                
                val_loss += loss.item()
                batch_count += 1
        
        avg_val_loss = val_loss / batch_count
        self.val_losses.append(avg_val_loss)
        
        print(f"Current train loss mean: {np.mean(self.train_losses[-100:]):.4f}")
        print(f"Current validation loss: {avg_val_loss:.4f}")
        
        return avg_val_loss
    
    def plot_training_progress(self):
        plt.figure(figsize=(12, 4))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        train_indices = np.arange(len(self.train_losses)) * 100  # Multiply by 100 as we log every 100 batches
        plt.plot(train_indices, self.train_losses, 'b.-', label='Training Loss', alpha=0.6)
        plt.title('Training Loss')
        plt.xlabel('Batch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        
        # Plot validation loss
        if self.val_losses:
            plt.subplot(1, 2, 2)
            val_indices = range(len(self.val_losses))
            plt.plot(val_indices, self.val_losses, 'r.-', label='Validation Loss', alpha=0.6)
            plt.title('Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        
        # Ensure directory exists
        os.makedirs('plots', exist_ok=True)
        
        # Save with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(f'plots/training_progress_{timestamp}.png')
        plt.close()
        
        # Save losses to separate CSVs to avoid length mismatch
        train_df = pd.DataFrame({
            'batch': range(len(self.train_losses)),
            'train_loss': self.train_losses
        })
        train_df.to_csv(f'plots/training_losses_{timestamp}.csv', index=False)
        
        if self.val_losses:
            val_df = pd.DataFrame({
                'epoch': range(len(self.val_losses)),
                'val_loss': self.val_losses
            })
            val_df.to_csv(f'plots/validation_losses_{timestamp}.csv', index=False)
    
    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, path)
    
    def train(self, num_epochs, save_path='checkpoints'):
        os.makedirs(save_path, exist_ok=True)
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            print(f"Training Loss: {train_loss:.4f}")
            
            val_loss = self.validate()
            if val_loss is not None:
                print(f"Validation Loss: {val_loss:.4f}")
                self.scheduler.step(val_loss)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(f'{save_path}/best_model.pt')
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    print("Early stopping triggered!")
                    break
            
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'{save_path}/checkpoint_epoch_{epoch+1}.pt')
            
            self.plot_training_progress()

def train_model(data_path, batch_size=32, num_epochs=10, lr=1e-3):
    # Define feature dimensions based on our dataset structure
    feature_dims = {
        'ttc': 2,
        'acc_info': 4,
        'dynamics': 3,
        'surroundings': 8,
        'lane_change': 4,  # New lane change features
        'overtaking': 4    # New overtaking features
    }
    
    print("Initializing dataset...")
    # Create dataset and split into train/val
    full_dataset = TrajectoryDataset(data_path, num_trajectories=10)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")
    
    # Create train/val splits
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    print("Creating data loaders...")
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        drop_last=True  # Ensure consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    
    print("Initializing model...")
    # Initialize model with fixed seed for reproducibility
    torch.manual_seed(42)
    model = MaxEntIRLNet(feature_dims)
    
    print("Setting up training manager...")
    # Initialize training manager
    trainer = TrainingManager(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr
    )
    
    print("Starting training...")
    # Train model
    trainer.train(num_epochs)
    
    return model, trainer

def load_model(model_path, feature_dims):
    """Utility function to load a trained model"""
    model = MaxEntIRLNet(feature_dims)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Training configuration
    config = {
        'data_path': "training_data.csv",  # Path to your CSV file
        'batch_size': 128,        # Increased batch size
        'num_epochs': 30,        # More training epochs
        'learning_rate': 5e-4,   # Lower learning rate
        'weight_decay': 5e-4     # Increased regularization
    }
    
    print("Starting MaxEnt IRL training...")
    print("Configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    try:
        # Train model
        model, trainer = train_model(
            data_path=config['data_path'],
            batch_size=config['batch_size'],
            num_epochs=config['num_epochs'],
            lr=config['learning_rate']
        )
        
        # Save final model
        final_path = 'final_model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_dims': {
                'ttc': 2,
                'acc_info': 4,
                'dynamics': 3,
                'surroundings': 8,
                'lane_change': 4,  # New lane change features
                'overtaking': 4    # New overtaking features
            },
            'config': config
        }, final_path)

        print(f"\nTraining completed successfully!")
        print(f"Final model saved to: {final_path}")
        
        # Plot final training curves
        trainer.plot_training_progress()
        print("Training plots saved to: training_progress.png")
        
    except Exception as e:
        print(f"\nAn error occurred during training: {str(e)}")
        raise