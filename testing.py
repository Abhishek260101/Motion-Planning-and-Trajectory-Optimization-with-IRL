import carla
import torch
import torch.nn as nn
import numpy as np
import time
import random
import math
from typing import List, Tuple

class SpectatorConfig:
    def __init__(self):
        self.height = 50.0        # Height above vehicle
        self.distance = 20.0      # Distance behind vehicle
        self.pitch = -45.0        # Camera angle down
        self.smooth_factor = 0.1  # Smoothing for camera movement

class HighwaySection:
    def __init__(self):
        self.x_min = -362
        self.x_max = 350
        self.y_min = 4
        self.y_max = 45

class TrajectoryGenerator:
    def __init__(self, n_trajectories=5, dt=0.2, horizon=6.0):
        self.n_trajectories = n_trajectories
        self.dt = dt
        self.horizon = horizon
        self.n_steps = int(horizon / dt)
        
    def generate_trajectories(self, current_state: dict) -> torch.Tensor:
        trajectories = []
        accelerations = np.linspace(-4.0, 1.5, self.n_trajectories)
        
        for acc in accelerations:
            trajectory = []
            x = float(current_state['Position_X'])
            y = float(current_state['Position_Y'])
            vx = float(current_state['Velocity_X'])
            vy = float(current_state['Velocity_Y'])
            
            for t in np.arange(0, self.horizon, self.dt):
                x_new = x + vx * t + 0.5 * acc * t * t
                y_new = y + vy * t
                vx_new = vx + acc * t
                
                trajectory.append([x_new, y_new, vx_new, vy, acc])
            
            trajectories.append(trajectory)
        
        return torch.tensor(trajectories, dtype=torch.float32)

class MaxEntIRLModel(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, sequence_length=30):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length
        
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x shape: [batch_size, n_trajectories, sequence_length, features]
        batch_size, n_trajectories, seq_len, feat_dim = x.shape
        
        # Reshape for feature extraction
        x = x.view(-1, feat_dim)  # Combine all dimensions except features
        x = self.feature_net(x)
        
        # Reshape for LSTM
        x = x.view(batch_size * n_trajectories, seq_len, self.hidden_dim)
        lstm_out, _ = self.lstm(x)
        
        # Use last LSTM output
        last_output = lstm_out[:, -1]
        
        # Final prediction
        x = self.fc(last_output)
        x = x.view(batch_size, n_trajectories)
        
        return x

def apply_vehicle_control(ego_vehicle, trajectory, world):
    """Apply control to ego vehicle with lane changing"""
    if not ego_vehicle or not ego_vehicle.is_alive:
        return
    
    # Check if lane change is needed
    change_lane, distance_to_vehicle = should_change_lane(ego_vehicle, world)
    
    # Get current waypoint
    current_waypoint = world.get_map().get_waypoint(ego_vehicle.get_location())
    
    # Create control
    control = carla.VehicleControl()
    
    if change_lane:
        # Check for possible lane changes
        right_lane = current_waypoint.get_right_lane()
        left_lane = current_waypoint.get_left_lane()
        
        # Prefer right lane change first if available
        if right_lane and right_lane.lane_type == carla.LaneType.Driving:
            control.steer = 0.3  # Right lane change
            print("Changing lane right")
        elif left_lane and left_lane.lane_type == carla.LaneType.Driving:
            control.steer = -0.3  # Left lane change
            print("Changing lane left")
        else:
            # No lane change possible, slow down
            control.brake = 0.5
            print("No lane change possible, slowing down")
    else:
        # Normal driving
        target_speed = trajectory[0][2]  # vx from trajectory
        current_speed = math.sqrt(
            ego_vehicle.get_velocity().x ** 2 + 
            ego_vehicle.get_velocity().y ** 2
        )
        
        # Speed control
        speed_diff = target_speed - current_speed
        control.throttle = max(0, min(1, speed_diff / 10.0))
        control.brake = max(0, min(1, -speed_diff / 10.0))
        
        # Lane following
        next_waypoint = current_waypoint.next(2.0)[0]
        if next_waypoint:
            direction = next_waypoint.transform.get_forward_vector()
            ego_direction = ego_vehicle.get_transform().get_forward_vector()
            
            cross_product = ego_direction.x * direction.y - ego_direction.y * direction.x
            control.steer = max(-1.0, min(1.0, cross_product))
    
    ego_vehicle.apply_control(control)

def cleanup(world, vehicles_list, ego_vehicle):
    """Cleanup function to reset the simulation"""
    try:
        # Reset world settings if world exists
        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
        
        # Destroy all spawned vehicles
        if vehicles_list:
            for vehicle in vehicles_list:
                if vehicle and vehicle.is_alive:
                    vehicle.destroy()
        
        # Destroy ego vehicle
        if ego_vehicle and ego_vehicle.is_alive:
            ego_vehicle.destroy()
            
    except Exception as e:
        print(f"Error during cleanup: {e}")


def update_spectator(world, ego_vehicle, config):
    """Update spectator position to follow ego vehicle"""
    ego_transform = ego_vehicle.get_transform()
    ego_location = ego_transform.location
    ego_rotation = ego_transform.rotation
    
    # Calculate spectator position
    spectator_location = carla.Location(
        x=ego_location.x - config.distance * math.cos(math.radians(ego_rotation.yaw)),
        y=ego_location.y - config.distance * math.sin(math.radians(ego_rotation.yaw)),
        z=ego_location.z + config.height
    )
    
    spectator_rotation = carla.Rotation(
        pitch=config.pitch,
        yaw=ego_rotation.yaw,
        roll=0.0
    )
    
    world.get_spectator().set_transform(
        carla.Transform(spectator_location, spectator_rotation)
    )

def connect_to_carla(retries=3, timeout=20.0):
    """Connect to CARLA with retry logic"""
    for attempt in range(retries):
        try:
            print(f"Attempting to connect to CARLA (Attempt {attempt + 1}/{retries})")
            client = carla.Client('localhost', 2000)
            client.set_timeout(timeout)
            
            # Test connection by getting world
            world = client.get_world()
            print("Successfully connected to CARLA")
            return client, world
            
        except Exception as e:
            print(f"Connection attempt {attempt + 1} failed: {str(e)}")
            if attempt < retries - 1:
                print("Retrying in 2 seconds...")
                time.sleep(2)
    
    raise ConnectionError("Failed to connect to CARLA after multiple attempts")

def spawn_ego_vehicle(world, highway_section):
    """Spawn ego vehicle near the starting edge with multiple attempts"""
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('role_name', 'ego')
    
    # Define several potential spawn positions near start
    spawn_positions = [
        {'x': -360, 'y': 40, 'z': 1},
        {'x': -360, 'y': 30, 'z': 1},
        {'x': -360, 'y': 30, 'z': 1},
        {'x': -355, 'y': 25, 'z': 1},
        {'x': -355, 'y': 25, 'z': 1},
        {'x': -350, 'y': 20, 'z': 1}
    ]
    
    # Try each spawn position
    for pos in spawn_positions:
        try:
            spawn_transform = carla.Transform()
            spawn_transform.location = carla.Location(x=pos['x'], y=pos['y'], z=pos['z'])
            spawn_transform.rotation = carla.Rotation(pitch=0, yaw=0, roll=0)
            
            # Check if location is clear before spawning
            if is_spawn_point_clear(world, spawn_transform.location):
                ego_vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
                if ego_vehicle:
                    world.tick()
                    ego_vehicle.enable_constant_velocity(carla.Vector3D(8, 0, 0))
                    print(f"Spawned ego vehicle with ID: {ego_vehicle.id} at x={spawn_transform.location.x:.1f}, y={spawn_transform.location.y:.1f}")
                    return ego_vehicle
                    
        except Exception as e:
            print(f"Failed spawn attempt at position {pos}: {e}")
            continue
        
    print("Failed to spawn ego vehicle after trying all positions")
    return None

def is_spawn_point_clear(world, location, min_distance=3.0):
    """Check if a spawn point is clear of obstacles"""
    
    # Get all actors in the world
    all_actors = world.get_actors()
    vehicles = all_actors.filter('vehicle.*')
    
    # Check distance to all vehicles
    for vehicle in vehicles:
        distance = location.distance(vehicle.get_location())
        if distance < min_distance:
            return False
            
    return True
def is_in_area_of_interest(location, highway_section):
    """Check if a location is within our area of interest"""
    return (highway_section.x_min <= location.x <= highway_section.x_max and
            highway_section.y_min <= location.y <= highway_section.y_max)

def get_random_spawn_location(world, highway_section):
    """Get a random spawn location in the middle of the area"""
    # Buffer from edges
    buffer = 50
    x_min = highway_section.x_min + buffer
    x_max = highway_section.x_max - buffer
    
    waypoints = world.get_map().generate_waypoints(2.0)
    valid_waypoints = [
        wp for wp in waypoints
        if (x_min <= wp.transform.location.x <= x_max and
            highway_section.y_min <= wp.transform.location.y <= highway_section.y_max)
    ]
    
    if valid_waypoints:
        waypoint = random.choice(valid_waypoints)
        transform = waypoint.transform
        # Add random lane offset
        transform.location.y += random.choice([-3.5, 0, 3.5])
        transform.location.z += 0.5  # Slight lift to avoid collision
        return transform
    return None

def monitor_and_respawn_traffic(world, vehicles_list, highway_section, traffic_manager):
    """Monitor traffic vehicles and respawn those that leave the area"""
    for vehicle in vehicles_list[:]:  # Use slice to avoid modifying list while iterating
        if vehicle.is_alive:
            location = vehicle.get_location()
            
            # Check if vehicle is outside area of interest
            if not is_in_area_of_interest(location, highway_section):
                print(f"Vehicle {vehicle.id} left area of interest, respawning...")
                
                # Get new spawn location
                new_spawn = get_random_spawn_location(world, highway_section)
                if new_spawn:
                    try:
                        # Keep same autopilot and speed settings
                        vehicle.set_transform(new_spawn)
                        print(f"Respawned vehicle {vehicle.id} at x={new_spawn.location.x:.1f}")
                    except Exception as e:
                        print(f"Failed to respawn vehicle {vehicle.id}: {e}")
                        # If respawn fails, destroy and create new vehicle
                        vehicle.destroy()
                        vehicles_list.remove(vehicle)
                        
                        # Try to spawn new vehicle
                        blueprint = random.choice(world.get_blueprint_library().filter('vehicle.*'))
                        try:
                            new_vehicle = world.spawn_actor(blueprint, new_spawn)
                            if new_vehicle:
                                new_vehicle.set_autopilot(True, traffic_manager.get_port())
                                traffic_manager.vehicle_percentage_speed_difference(
                                    new_vehicle, 
                                    random.uniform(40, 60)
                                )
                                vehicles_list.append(new_vehicle)
                                print(f"Created new vehicle {new_vehicle.id}")
                        except Exception as spawn_error:
                            print(f"Failed to create new vehicle: {spawn_error}")

def setup_traffic_vehicles(client, world, highway_section, num_vehicles=50):
    """Spawn traffic vehicles in the middle section"""
    vehicles_list = []
    
    try:
        blueprint_library = world.get_blueprint_library()
        
        # Generate waypoints instead of using spawn points
        waypoints = world.get_map().generate_waypoints(10.0)  # Every 10 meters
        
        # Filter waypoints for middle section (avoid first and last 50 meters)
        valid_waypoints = [
            wp for wp in waypoints
            if (-312 <= wp.transform.location.x <= 300 and  # Excluded 50m from each end
                4 <= wp.transform.location.y <= 45)
        ]
        
        if not valid_waypoints:
            print("No valid waypoints in middle section!")
            return vehicles_list
            
        # Setup traffic manager
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        traffic_manager.set_global_distance_to_leading_vehicle(5.0)
        traffic_manager.global_percentage_speed_difference(40)
        
        # Sort waypoints by x coordinate
        valid_waypoints.sort(key=lambda wp: wp.transform.location.x)
        waypoint_step = len(valid_waypoints) // num_vehicles
        
        for i in range(num_vehicles):
            blueprint = random.choice(blueprint_library.filter('vehicle.*'))
            if blueprint.has_attribute('role_name'):
                blueprint.set_attribute('role_name', 'autopilot')
            
            # Get waypoint based on index
            if i * waypoint_step < len(valid_waypoints):
                waypoint = valid_waypoints[i * waypoint_step]
                
                # Create spawn transform with slight y offset for different lanes
                spawn_transform = waypoint.transform
                spawn_transform.location.y += random.choice([-3.5, 0, 3.5])  # Different lanes
                spawn_transform.location.z += 0.5  # Slight lift to avoid collision
                
                try:
                    vehicle = world.spawn_actor(blueprint, spawn_transform)
                    if vehicle:
                        vehicle.set_autopilot(True, traffic_manager.get_port())
                        traffic_manager.vehicle_percentage_speed_difference(
                            vehicle, 
                            random.uniform(40, 60)
                        )
                        vehicles_list.append(vehicle)
                        print(f"Spawned vehicle {vehicle.id} at x={spawn_transform.location.x:.1f}")
                        world.tick()
                except Exception as e:
                    print(f"Failed to spawn vehicle at {spawn_transform.location.x:.1f}: {e}")
                    continue
        
        print(f"Successfully spawned {len(vehicles_list)} vehicles")
        return vehicles_list
        
    except Exception as e:
        print(f"Error in setup_traffic_vehicles: {e}")
        for vehicle in vehicles_list:
            if vehicle and vehicle.is_alive:
                vehicle.destroy()
        return []
        
def check_for_vehicles_ahead(ego_vehicle, world, distance_threshold=20.0):
    """Check for vehicles ahead of ego vehicle"""
    ego_location = ego_vehicle.get_location()
    ego_transform = ego_vehicle.get_transform()
    ego_forward = ego_transform.get_forward_vector()
    
    # Get all vehicles
    vehicles = world.get_actors().filter('vehicle.*')
    
    vehicles_ahead = []
    for vehicle in vehicles:
        if vehicle.id != ego_vehicle.id:
            vehicle_location = vehicle.get_location()
            
            # Calculate relative position
            dx = vehicle_location.x - ego_location.x
            dy = vehicle_location.y - ego_location.y
            
            # Check if vehicle is ahead
            forward_dot = dx * ego_forward.x + dy * ego_forward.y
            
            if forward_dot > 0:  # Vehicle is ahead
                distance = math.sqrt(dx*dx + dy*dy)
                if distance < distance_threshold:
                    vehicles_ahead.append((vehicle, distance))
    
    return vehicles_ahead

def should_change_lane(ego_vehicle, world):
    """Determine if lane change is needed"""
    vehicles_ahead = check_for_vehicles_ahead(ego_vehicle, world)
    
    if vehicles_ahead:
        nearest_vehicle, distance = min(vehicles_ahead, key=lambda x: x[1])
        
        # Get ego vehicle speed
        ego_vel = ego_vehicle.get_velocity()
        ego_speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2)
        
        # Get lead vehicle speed
        lead_vel = nearest_vehicle.get_velocity()
        lead_speed = math.sqrt(lead_vel.x**2 + lead_vel.y**2)
        
        # Change lane if lead vehicle is slower and close
        if lead_speed < ego_speed and distance < 15.0:
            return True, distance
    
    return False, float('inf')

def prepare_model_input(current_state, trajectory_generator):
    """Prepare input tensor for the model with correct dimensions"""
    # Generate trajectories
    trajectories = trajectory_generator.generate_trajectories(current_state)
    
    # Reshape for model: [batch_size, sequence_length, features]
    # Assuming trajectories shape is [n_trajectories, n_steps, features]
    trajectories = trajectories.unsqueeze(0)  # Add batch dimension
    
    return trajectories
def is_in_highway_section(location, highway_section):
    """Check if a location is within the highway section"""
    return (highway_section.x_min <= location.x <= highway_section.x_max and
            highway_section.y_min <= location.y <= highway_section.y_max)

def main():
    vehicles_list = []
    ego_vehicle = None
    world = None
    
    try:
        # Connect to CARLA
        client, world = connect_to_carla()
        
        # Load Town04
        print("Loading Town04...")
        world = client.load_world('Town04')
        time.sleep(2.0)
        
        # Setup
        highway_section = HighwaySection()
        spectator_config = SpectatorConfig()
        
        # Set synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Initialize model
        print("Loading IRL model...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MaxEntIRLModel().to(device)
        model.load_state_dict(torch.load('irl_model.pth'))
        model.eval()
        
        # Setup traffic and traffic manager
        print("Setting up traffic...")
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)
        vehicles_list = setup_traffic_vehicles(client, world, highway_section)
        
        if len(vehicles_list) == 0:
            print("Warning: No traffic vehicles spawned!")
        
        # Spawn ego vehicle
        max_attempts = 3
        ego_vehicle = None
        
        for attempt in range(max_attempts):
            print(f"Attempting to spawn ego vehicle (Attempt {attempt + 1}/{max_attempts})")
            ego_vehicle = spawn_ego_vehicle(world, highway_section)
            if ego_vehicle:
                break
            else:
                print("Retrying ego vehicle spawn...")
                time.sleep(1)  # Wait before retry
        
        if not ego_vehicle:
            raise RuntimeError("Failed to spawn ego vehicle after multiple attempts")
        
        # Initialize trajectory generator
        trajectory_generator = TrajectoryGenerator()
        
        print("Starting simulation...")
        while True:
            world.tick()
            
            # Monitor and respawn traffic vehicles that left the area
            monitor_and_respawn_traffic(world, vehicles_list, highway_section, traffic_manager)
            
            # Update spectator
            if ego_vehicle and ego_vehicle.is_alive:
                update_spectator(world, ego_vehicle, spectator_config)
                
                # Get current state
                location = ego_vehicle.get_location()
                velocity = ego_vehicle.get_velocity()
                acceleration = ego_vehicle.get_acceleration()
                
                current_state = {
                    'Position_X': location.x,
                    'Position_Y': location.y,
                    'Velocity_X': velocity.x,
                    'Velocity_Y': velocity.y,
                    'Acceleration_X': acceleration.x
                }
                
                # Prepare input for model
                trajectories = prepare_model_input(current_state, trajectory_generator)
                trajectories = trajectories.to(device)
                
                # Get model prediction and apply control
                with torch.no_grad():
                    scores = model(trajectories)
                    best_trajectory_idx = torch.argmax(scores)
                    best_trajectory = trajectories[0, best_trajectory_idx].cpu().numpy()
                
                # Apply control with lane changing behavior
                apply_vehicle_control(ego_vehicle, best_trajectory, world)
                
                # Print state with additional info
                print(f"\nEgo Vehicle State:")
                print(f"Position: ({location.x:.2f}, {location.y:.2f})")
                print(f"Velocity: ({velocity.x:.2f}, {velocity.y:.2f})")
                print(f"Selected trajectory: {best_trajectory_idx.item()}")
                
                # Print traffic info
                vehicles_ahead = check_for_vehicles_ahead(ego_vehicle, world)
                if vehicles_ahead:
                    print(f"Vehicles ahead: {len(vehicles_ahead)}")
                    print(f"Nearest vehicle distance: {vehicles_ahead[0][1]:.2f}m")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("\nCleaning up...")
        cleanup(world, vehicles_list, ego_vehicle)
        print("Cleanup complete!")

if __name__ == '__main__':
    main()
