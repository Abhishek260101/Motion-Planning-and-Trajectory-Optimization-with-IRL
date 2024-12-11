import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

class IRLTrainer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.feature_dim = 5  # [pos_x, pos_y, velocity, steering, distance_to_target, is_turning]
        
        # Model parameters
        self.hidden_layers = [128, 64, 32]
        self.learning_rate = 0.001
        
        # Initialize reward network
        self.reward_network = self._build_reward_network()
        
    def _build_reward_network(self):
        """Build neural network for reward function"""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.feature_dim,)))
        
        # Hidden layers
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(0.2))
        
        # Output layer (single reward value)
        model.add(layers.Dense(1))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.maximum_entropy_loss
        )
        
        return model
    
    def load_data(self):
        """Load and preprocess the driving data"""
        # Load CSV data
        csv_path = os.path.join(self.data_dir, 'driving_data.csv')
        df = pd.read_csv(csv_path)
        
        # Extract features
        features = np.column_stack([
            df['pos_x'].values,
            df['pos_y'].values,
            df['velocity'].values,
            df['steering'].values,
            df['distance_to_target'].values
            # Removed turn_direction
        ])
        
        # Normalize features
        self.feature_means = np.mean(features, axis=0)
        self.feature_stds = np.std(features, axis=0)
        normalized_features = (features - self.feature_means) / (self.feature_stds + 1e-8)
        
        return normalized_features
    
    def maximum_entropy_loss(self, expert_features, predicted_rewards):
        """
        Maximum entropy IRL loss function
        Based on Ziebart et al. 2008 paper
        """
        # Expert trajectory likelihood
        expert_likelihood = tf.reduce_mean(predicted_rewards)
        
        # Partition function (normalize over all possible trajectories)
        # In practice, we use the batch as an approximation
        partition = tf.reduce_logsumexp(predicted_rewards)
        
        # Maximum entropy loss
        loss = -(expert_likelihood - partition)
        
        return loss
    
    def train(self, epochs=100, batch_size=32):
        """Train the IRL model"""
        # Load and preprocess data
        features = self.load_data()
        
        # Split data
        train_features, val_features = train_test_split(
            features, test_size=0.2, random_state=42
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': []
        }
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle training data
            np.random.shuffle(train_features)
            
            # Train on batches
            train_losses = []
            for i in range(0, len(train_features), batch_size):
                batch = train_features[i:i + batch_size]
                loss = self.reward_network.train_on_batch(batch, np.zeros((len(batch), 1)))
                train_losses.append(loss)
            
            # Validation
            val_loss = self.reward_network.evaluate(
                val_features, 
                np.zeros((len(val_features), 1)), 
                verbose=0
            )
            
            # Record history
            history['train_loss'].append(np.mean(train_losses))
            history['val_loss'].append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {history['train_loss'][-1]:.4f}")
                print(f"Val Loss: {history['val_loss'][-1]:.4f}")
        
        return history
    
    def save_model(self, path):
        """Save the trained reward network"""
        self.reward_network.save(path)
        
        # Save normalization parameters
        np.savez(
            path + '_norm_params.npz',
            means=self.feature_means,
            stds=self.feature_stds
        )
    
    def plot_training_history(self, history):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('IRL Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.data_dir, 'training_history.png'))
        plt.close()

def main():
    # Find most recent data directory
    base_dir = "hybrid_data_"
    data_dirs = [d for d in os.listdir() if d.startswith(base_dir)]
    if not data_dirs:
        raise Exception("No data directory found")
    
    latest_dir = max(data_dirs)
    print(f"Using data from: {latest_dir}")
    
    # Initialize and train IRL
    trainer = IRLTrainer(latest_dir)
    
    print("Starting IRL training...")
    history = trainer.train(epochs=100)
    
    # Save results
    trainer.save_model(os.path.join(latest_dir, 'irl_model'))
    trainer.plot_training_history(history)
    print("Training complete!")

if __name__ == "__main__":
    main()
