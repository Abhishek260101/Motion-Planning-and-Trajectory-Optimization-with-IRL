import numpy as np
import carla
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import time
from tensorflow.keras.utils import custom_object_scope

def maximum_entropy_loss(expert_features, predicted_rewards):
    """
    Maximum entropy IRL loss function
    Based on Ziebart et al. 2008 paper
    """
    # Expert trajectory likelihood
    expert_likelihood = tf.reduce_mean(predicted_rewards)
    
    # Partition function
    partition = tf.reduce_logsumexp(predicted_rewards)
    
    # Maximum entropy loss
    loss = -(expert_likelihood - partition)
    
    return loss

class IRLTester:
    def __init__(self, model_path, data_dir):
        # Load trained model with custom loss function
        with custom_object_scope({'maximum_entropy_loss': maximum_entropy_loss}):
            self.reward_model = load_model(model_path)
            
        norm_params = np.load(model_path + '_norm_params.npz')
        self.feature_means = norm_params['means']
        self.feature_stds = norm_params['stds']
        
        # CARLA setup
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Vehicle setup
        self.vehicle = None
        self.sensor_data = {}
        
    def setup_vehicle(self):
        """Setup test vehicle"""
        try:
            # Get random spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = np.random.choice(spawn_points)
            
            # Spawn vehicle
            blueprint = self.world.get_blueprint_library().filter('model3')[0]
            self.vehicle = self.world.spawn_actor(blueprint, spawn_point)
            
            print("Vehicle spawned successfully")
            return True
        except Exception as e:
            print(f"Error setting up vehicle: {e}")
            return False
            
    def get_state_features(self):
        """Get current state features"""
        if not self.vehicle:
            return None
            
        # Get vehicle state
        transform = self.vehicle.get_transform()
        velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        
        # Get current waypoint
        waypoint = self.world.get_map().get_waypoint(transform.location)
        
        # Create feature vector (5 features instead of 6)
        features = np.array([
            transform.location.x,
            transform.location.y,
            speed,
            0.0,  # steering will be filled for each action
            waypoint.transform.location.distance(transform.location)
            # Removed turning flag
        ])
        
        return features
    
    def normalize_features(self, features):
        """Normalize features using saved parameters"""
        return (features - self.feature_means) / (self.feature_stds + 1e-8)
    
    def get_best_action(self):
        """Get best action based on learned reward"""
        if not self.vehicle:
            return None
            
        base_features = self.get_state_features()
        if base_features is None:
            return None
            
        # Possible steering angles
        steering_angles = np.linspace(-1.0, 1.0, 21)  # 21 possible angles
        best_reward = float('-inf')
        best_action = None
        
        # Evaluate each possible action
        for steering in steering_angles:
            features = base_features.copy()
            features[3] = steering  # Set steering
            # Removed setting turning flag since we removed that feature
            
            # Normalize and predict reward
            norm_features = self.normalize_features(features)
            reward = self.reward_model.predict(norm_features.reshape(1, -1), verbose=0)[0][0]
            
            if reward > best_reward:
                best_reward = reward
                best_action = steering
        
        return best_action
    
    def test_model(self, duration=300):  # 5 minutes test
        """Test the learned model"""
        try:
            if not self.setup_vehicle():
                return
            
            print("Starting test drive...")
            start_time = time.time()
            
            while time.time() - start_time < duration:
                # Get best action from learned model
                steering = self.get_best_action()
                if steering is None:
                    continue
                
                # Get speed based on steering (slow down for turns)
                speed = 30.0 * (1.0 - 0.5 * abs(steering))  # Reduce speed in turns
                current_speed = np.sqrt(sum(x*x for x in [self.vehicle.get_velocity().x, 
                                                        self.vehicle.get_velocity().y]))
                
                # Calculate throttle/brake
                if current_speed < speed:
                    throttle = 0.7
                    brake = 0.0
                else:
                    throttle = 0.0
                    brake = 0.3
                
                # Apply control
                control = carla.VehicleControl(
                    throttle=throttle,
                    steer=float(steering),
                    brake=brake
                )
                self.vehicle.apply_control(control)
                
                # Print current state (optional)
                if int(time.time()) % 5 == 0:  # Print every 5 seconds
                    print(f"Steering: {steering:.3f}, Speed: {current_speed:.2f}")
                
                time.sleep(0.05)  # Control frequency
                
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        except Exception as e:
            print(f"Error during test: {e}")
        finally:
            if self.vehicle:
                self.vehicle.destroy()
                
def main():
    # Find most recent data directory
    base_dir = "hybrid_data_"
    data_dirs = [d for d in os.listdir() if d.startswith(base_dir)]
    if not data_dirs:
        raise Exception("No data directory found")
    
    latest_dir = max(data_dirs)
    model_path = os.path.join(latest_dir, 'irl_model')
    
    if not os.path.exists(model_path):
        raise Exception("Trained model not found")
    
    print(f"Using model from: {model_path}")
    
    # Initialize and run test
    tester = IRLTester(model_path, latest_dir)
    tester.test_model()

if __name__ == "__main__":
    main()