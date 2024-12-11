import carla
import random
import time
import numpy as np
import csv
from datetime import datetime
import os
import math

class HighwaySection:
    def __init__(self):
        self.x_min = -362
        self.x_max = 350
        self.y_min = 4
        self.y_max = 45
        self.width = self.x_max - self.x_min
        self.length = self.y_max - self.y_min

def setup_csv_file():
    """Create and setup CSV file with enhanced headers"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'vehicle_data_{timestamp}.csv'
    headers = [
        'Timestamp',
        'Vehicle_ID',
        'Position_X',
        'Position_Y',
        'Velocity_X',
        'Velocity_Y',
        'Acceleration_X',
        'Acceleration_Y',
        'Lane_ID',
        'Left_Lane_Vehicles',
        'Right_Lane_Vehicles',
        'Time_To_Collision',
        'Distance_To_Front_Vehicle',
        'Front_Vehicle_Speed',
        'Distance_To_Left_Vehicle',
        'Distance_To_Right_Vehicle',
        'Lane_Change_State',  # none, left, right
        'Lane_Change_Progress',  # 0 to 1
        'Is_Safe_Left_Lane',  # boolean
        'Is_Safe_Right_Lane',  # boolean
        'Relative_Speed_To_Front',  # speed difference with front vehicle
        'Current_Lane_Speed',  # average speed in current lane
        'Left_Lane_Speed',    # average speed in left lane
        'Right_Lane_Speed'    # average speed in right lane
    ]
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    return filename

def get_lane_info(vehicle, vehicles, world, max_distance=50.0):
    """Get comprehensive lane and surrounding vehicle information"""
    vehicle_loc = vehicle.get_location()
    vehicle_wp = world.get_map().get_waypoint(vehicle_loc)
    vehicle_speed = math.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2)
    
    # Initialize data
    info = {
        'front_distance': float('inf'),
        'front_speed': 0,
        'left_distance': float('inf'),
        'right_distance': float('inf'),
        'is_safe_left': True,
        'is_safe_right': True,
        'current_lane_speed': vehicle_speed,
        'left_lane_speed': 0,
        'right_lane_speed': 0,
        'relative_front_speed': 0
    }
    
    # Check each vehicle
    for other in vehicles:
        if other.id == vehicle.id:
            continue
            
        other_loc = other.get_location()
        other_wp = world.get_map().get_waypoint(other_loc)
        other_speed = math.sqrt(other.get_velocity().x**2 + other.get_velocity().y**2)
        
        # Calculate distance and relative position
        distance = vehicle_loc.distance(other_loc)
        if distance > max_distance:
            continue
            
        # Get relative position
        rel_pos = other_loc - vehicle_loc
        forward_vec = vehicle.get_transform().get_forward_vector()
        right_vec = carla.Vector3D(x=-forward_vec.y, y=forward_vec.x, z=0)
        
        forward_dot = rel_pos.x * forward_vec.x + rel_pos.y * forward_vec.y
        right_dot = rel_pos.x * right_vec.x + rel_pos.y * right_vec.y
        
        # Same lane - check if vehicle is in front
        if other_wp.lane_id == vehicle_wp.lane_id and forward_dot > 0:
            if distance < info['front_distance']:
                info['front_distance'] = distance
                info['front_speed'] = other_speed
                info['relative_front_speed'] = other_speed - vehicle_speed
        
        # Left lane
        elif vehicle_wp.get_left_lane() and other_wp.lane_id == vehicle_wp.get_left_lane().lane_id:
            info['left_distance'] = min(info['left_distance'], distance)
            info['left_lane_speed'] += other_speed
            if distance < 20.0:  # Safety threshold
                info['is_safe_left'] = False
        
        # Right lane
        elif vehicle_wp.get_right_lane() and other_wp.lane_id == vehicle_wp.get_right_lane().lane_id:
            info['right_distance'] = min(info['right_distance'], distance)
            info['right_lane_speed'] += other_speed
            if distance < 20.0:  # Safety threshold
                info['is_safe_right'] = False
    
    return info

def create_spawn_points(world, highway_section):
    """Create spawn points within the highway section"""
    spawn_points = []
    map = world.get_map()
    waypoints = map.generate_waypoints(2.0)  # Generate waypoints every 2 meters
    
    for waypoint in waypoints:
        if is_within_highway(waypoint.transform.location, highway_section):
            transform = waypoint.transform
            # Add slight variations for different lanes
            for lane_offset in [-3.5, 0, 3.5]:  # Left lane, center, right lane
                new_transform = carla.Transform(
                    carla.Location(
                        x=transform.location.x,
                        y=transform.location.y + lane_offset,
                        z=transform.location.z + 0.5  # Lift slightly to avoid collisions
                    ),
                    transform.rotation
                )
                spawn_points.append(new_transform)
    
    return spawn_points

def is_within_highway(location, highway_section):
    """Check if a location is within the highway section"""
    return (highway_section.x_min <= location.x <= highway_section.x_max and 
            highway_section.y_min <= location.y <= highway_section.y_max)

def connect_to_carla():
    """Connect to CARLA server with retry logic"""
    for retry in range(3):
        try:
            print(f"Attempting to connect to CARLA (Attempt {retry + 1}/3)")
            client = carla.Client('localhost', 2000)
            client.set_timeout(20.0)
            world = client.get_world()
            print("Successfully connected to CARLA!")
            return client, world
        except Exception as e:
            print(f"Connection attempt {retry + 1} failed: {str(e)}")
            if retry < 2:
                print("Retrying in 5 seconds...")
                time.sleep(5)
    return None, None

def setup_traffic_vehicles(client, world, highway_section):
    """Spawn traffic vehicles with varying behaviors"""
    vehicles_list = []
    blueprint_library = world.get_blueprint_library()
    
    # Set up traffic manager
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(3.0)
    
    # Get spawn points
    spawn_points = create_spawn_points(world, highway_section)
    
    # Spawn vehicles with varying behaviors
    for i in range(50):
        vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
        spawn_point = random.choice(spawn_points)
        
        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            if vehicle:
                vehicle.set_autopilot(True, traffic_manager.get_port())
                
                # Randomize vehicle behavior
                speed_factor = random.uniform(-30, 10)
                lane_change_factor = random.uniform(30, 70)
                
                traffic_manager.vehicle_percentage_speed_difference(vehicle, speed_factor)
                traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(2, 5))
                traffic_manager.random_left_lanechange_percentage(vehicle, lane_change_factor)
                traffic_manager.random_right_lanechange_percentage(vehicle, lane_change_factor)
                
                vehicles_list.append(vehicle)
                print(f"Spawned vehicle {vehicle.id} with speed factor {speed_factor:.1f}%")
        
        except Exception as e:
            print(f"Failed to spawn vehicle: {e}")
    
    return vehicles_list

def main():
    vehicles = []
    world = None
    highway = HighwaySection()
    
    try:
        # Connect to CARLA and setup
        client, world = connect_to_carla()
        if not client or not world:
            return
        
        print("Loading Town04...")
        world = client.load_world('Town04')
        
        # Setup simulation
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Create traffic
        print("Setting up traffic...")
        vehicles = setup_traffic_vehicles(client, world, highway)
        
        if not vehicles:
            raise RuntimeError("No vehicles spawned!")
        
        # Setup data collection
        csv_filename = setup_csv_file()
        print(f"Logging data to: {csv_filename}")
        
        # Main data collection loop
        with open(csv_filename, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            while True:
                world.tick()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                for vehicle in vehicles:
                    try:
                        # Skip if vehicle is not valid
                        if not vehicle.is_alive:
                            continue
                            
                        # Get basic vehicle data
                        location = vehicle.get_location()
                        if not is_within_highway(location, highway):
                            new_spawn_point = random.choice(create_spawn_points(world, highway))
                            vehicle.set_transform(new_spawn_point)
                            continue
                        
                        # Collect comprehensive vehicle data
                        transform = vehicle.get_transform()
                        velocity = vehicle.get_velocity()
                        acceleration = vehicle.get_acceleration()
                        waypoint = world.get_map().get_waypoint(location)
                        
                        # Get lane information
                        lane_info = get_lane_info(vehicle, vehicles, world)
                        
                        # Get adjacent lane vehicles
                        left_lane_vehicles = []
                        right_lane_vehicles = []
                        if waypoint.get_left_lane():
                            left_lane_vehicles = [v.id for v in world.get_actors().filter('vehicle.*') 
                                            if world.get_map().get_waypoint(v.get_location()).lane_id == waypoint.get_left_lane().lane_id]
                        if waypoint.get_right_lane():
                            right_lane_vehicles = [v.id for v in world.get_actors().filter('vehicle.*') 
                                                if world.get_map().get_waypoint(v.get_location()).lane_id == waypoint.get_right_lane().lane_id]

                        # Write enhanced data to CSV
                        writer.writerow([
                            timestamp,
                            vehicle.id,
                            f"{location.x:.2f}",
                            f"{location.y:.2f}",
                            f"{velocity.x:.2f}",
                            f"{velocity.y:.2f}",
                            f"{acceleration.x:.2f}",
                            f"{acceleration.y:.2f}",
                            waypoint.lane_id,
                            ','.join(map(str, left_lane_vehicles)),
                            ','.join(map(str, right_lane_vehicles)),
                            f"{lane_info['front_distance']:.2f}",
                            f"{lane_info['front_speed']:.2f}",
                            f"{lane_info['left_distance']:.2f}",
                            f"{lane_info['right_distance']:.2f}",
                            'none',  # Lane change state
                            '0',     # Lane change progress
                            str(lane_info['is_safe_left']).lower(),
                            str(lane_info['is_safe_right']).lower(),
                            f"{lane_info['relative_front_speed']:.2f}",
                            f"{lane_info['current_lane_speed']:.2f}",
                            f"{lane_info['left_lane_speed']:.2f}",
                            f"{lane_info['right_lane_speed']:.2f}"
                        ])
                        
                        csv_file.flush()
                        
                    except Exception as e:
                        print(f"\nError processing vehicle {vehicle.id}: {e}")
                        continue

                
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nData collection interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if vehicles:
            for vehicle in vehicles:
                try:
                    vehicle.destroy()
                except:
                    pass
        if world:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)

if __name__ == '__main__':
    main()
