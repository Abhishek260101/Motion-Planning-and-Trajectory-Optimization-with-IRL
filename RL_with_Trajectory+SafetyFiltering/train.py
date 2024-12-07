from stable_baselines3 import PPO
import os
from environment import CarEnv
import time

def main():
    print('Setting up training environment...')
    
    # Create directories for logs and models
    timestamp = int(time.time())
    models_dir = f"models/{timestamp}/"
    logdir = f"logs/{timestamp}/"
    
    print(f"Models will be saved to: {models_dir}")
    print(f"Logs will be saved to: {logdir}")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created models directory: {models_dir}")
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)
        print(f"Created logs directory: {logdir}")
    
    # Create and configure environment
    env = CarEnv()
    print("Environment created successfully")
    
    # Initialize model
    print("Initializing PPO model...")
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        learning_rate=0.001,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=logdir
    )
    print("PPO model initialized")
    
    TIMESTEPS = 100_000  # Reduced for testing
    iters = 0
    
    try:
        while iters < 2:
            print(f'Starting iteration {iters + 1}...')
            start_time = time.time()
            
            # Train the model
            model.learn(
                total_timesteps=TIMESTEPS,
                reset_num_timesteps=False,
                tb_log_name=f"PPO_trajectory"
            )
            
            # Save the model
            save_path = f"{models_dir}/{TIMESTEPS * (iters + 1)}"
            model.save(save_path)
            print(f'Model saved to: {save_path}')
            
            end_time = time.time()
            duration = end_time - start_time
            print(f'Completed iteration {iters + 1} in {duration/60:.2f} minutes')
            
            iters += 1
            
    except Exception as e:
        print(f"Error during training: {e}")
        
    print("Training completed")

if __name__ == "__main__":
    main()