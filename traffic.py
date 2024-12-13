import carla
import random
import time
import numpy as np
import csv
from datetime import datetime
import os
import math

class HighwaySection:
    """Define area of interest for traffic"""
    def __init__(self):
        self.x_min = -362
        self.x_max = 350
        self.y_min = 4
        self.y_max = 45

class SimpleTrafficManager:
    """Simplified traffic management with collision handling and enhanced overtaking"""
    def __init__(self, client, world):
        self.client = client
        self.world = world
        self.tm = client.get_trafficmanager(8000)
        self.tm.set_synchronous_mode(True)
        
        # Define fixed speeds for each lane (km/h)
        self.lane_speeds = {
            -4: 60,   # Rightmost lane
            -3: 75,
            -2: 90,
            -1: 105   # Leftmost lane
        }
        
        self.vehicles = {}
        self.collision_sensors = {}
        self.highway_section = HighwaySection()
        
        # Behavior parameters
        self.MIN_DISTANCE = 15.0
        self.OVERTAKE_DISTANCE = 30.0  # Distance to check for overtaking
        self.SPEED_BOOST = 1.3  # 30% speed boost
        self.OVERTAKE_BOOST = 1.4  # 40% speed boost for overtaking
        self.SAFE_DISTANCE = 20.0

    def setup_vehicle(self, vehicle):
        """Setup vehicle and collision sensor"""
        vehicle.set_autopilot(True, self.tm.get_port())
        
        # Get current lane and assign speed
        wp = self.world.get_map().get_waypoint(vehicle.get_location())
        base_speed = self.lane_speeds.get(wp.lane_id, 75)
        speed_variation = random.uniform(-10, 10)
        target_speed = base_speed * (1 + speed_variation/100)
        
        # Convert to CARLA speed factor
        speed_factor = ((target_speed - 75) / 75) * 100
        self.tm.vehicle_percentage_speed_difference(vehicle, speed_factor)
        
        # Disable automatic lane changes
        self.tm.auto_lane_change(vehicle, False)
        self.tm.random_left_lanechange_percentage(vehicle, 0)
        self.tm.random_right_lanechange_percentage(vehicle, 0)
        
        # Setup collision sensor
        blueprint = self.world.get_blueprint_library().find('sensor.other.collision')
        sensor_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        sensor = self.world.spawn_actor(blueprint, sensor_transform, attach_to=vehicle)
        sensor.listen(lambda event: self._on_collision(event, vehicle))
        self.collision_sensors[vehicle.id] = sensor
        
        # Initialize vehicle state
        self.vehicles[vehicle.id] = {
            'base_speed': target_speed,
            'normal_factor': speed_factor,
            'changing_lane': False,
            'change_start': 0,
            'boosted': False,
            'is_overtaking': False
        }
        
        return speed_factor

    def _on_collision(self, event, vehicle):
        """Handle vehicle collision"""
        if vehicle.is_alive:
            if event.other_actor and isinstance(event.other_actor, carla.Vehicle):
                spawn_points = self._get_spawn_points()
                if spawn_points:
                    # Respawn both vehicles
                    vehicle.set_transform(random.choice(spawn_points))
                    if event.other_actor.is_alive:
                        event.other_actor.set_transform(random.choice(spawn_points))

    def _get_spawn_points(self):
        """Get valid spawn points in highway section"""
        spawn_points = []
        waypoints = self.world.get_map().generate_waypoints(2.0)
        
        for wp in waypoints:
            loc = wp.transform.location
            if (self.highway_section.x_min <= loc.x <= self.highway_section.x_max and 
                self.highway_section.y_min <= loc.y <= self.highway_section.y_max):
                transform = wp.transform
                transform.location.z += 0.5
                spawn_points.append(transform)
                
        return spawn_points

    def update_vehicle(self, vehicle):
        """Update vehicle behavior with enhanced overtaking and lane changing"""
        if vehicle.id not in self.vehicles:
            return
            
        state = self.vehicles[vehicle.id]
        
        # If currently changing lanes, check if complete
        if state['changing_lane']:
            if time.time() - state['change_start'] > 3.0:
                self._end_lane_change(vehicle)
            return
        
        # Get current speed and front vehicle info
        velocity = vehicle.get_velocity()
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2) * 3.6  # km/h
        front_info = self._get_front_vehicle_info(vehicle)
        
        # More aggressive lane change triggers
        if front_info['distance'] < self.OVERTAKE_DISTANCE:
            front_speed = front_info['speed'] * 3.6  # km/h
            
            # Check if front vehicle is stopped or very slow
            if front_speed < 5:  # Consider vehicle stopped if speed < 5 km/h
                self._attempt_overtake(vehicle, front_info, emergency=True)
                return
                
            # If front vehicle is significantly slower, attempt overtake
            if front_speed < current_speed - 5:
                self._attempt_overtake(vehicle, front_info)
                return
                
            # If too close to front vehicle, attempt emergency lane change
            if front_info['distance'] < self.MIN_DISTANCE:
                self._attempt_overtake(vehicle, front_info, emergency=True)
                return
        
        # Periodically check for better lanes even without immediate need
        if random.random() < 0.02:  # 2% chance each update
            self._check_for_better_lane(vehicle)

    def _check_for_better_lane(self, vehicle):
        """Check if a better lane is available and switch if beneficial"""
        wp = self.world.get_map().get_waypoint(vehicle.get_location())
        current_lane_id = wp.lane_id
        
        # Check left lane first (faster lane)
        left_lane = wp.get_left_lane()
        if left_lane and self._is_lane_better(vehicle, left_lane, 'left'):
            self._execute_lane_change(vehicle, 'left')
            return
            
        # Check right lane if left isn't better
        right_lane = wp.get_right_lane()
        if right_lane and self._is_lane_better(vehicle, right_lane, 'right'):
            self._execute_lane_change(vehicle, 'right')

    def _is_lane_better(self, vehicle, target_lane, direction):
        """Determine if target lane is better than current lane"""
        if not self._check_lane_safe(vehicle, target_lane):
            return False
        
        # Get current lane ID using world map
        current_wp = self.world.get_map().get_waypoint(vehicle.get_location())
        current_speed = self.lane_speeds.get(current_wp.lane_id, 75)
        target_speed = self.lane_speeds.get(target_lane.lane_id, 75)
        
        # Left lane should be faster
        if direction == 'left':
            return target_speed > current_speed
        
        # Right lane if current lane is too fast
        velocity = vehicle.get_velocity()
        actual_speed = math.sqrt(velocity.x**2 + velocity.y**2) * 3.6
        return actual_speed < current_speed - 15

    def _end_lane_change(self, vehicle):
        """Reset vehicle state after lane change"""
        state = self.vehicles[vehicle.id]
        state['changing_lane'] = False
        state['boosted'] = False
        state['is_overtaking'] = False
        self.tm.vehicle_percentage_speed_difference(vehicle, state['normal_factor'])

    def _get_front_vehicle_info(self, vehicle):
        """Get detailed information about vehicle in front"""
        loc = vehicle.get_location()
        fwd = vehicle.get_transform().get_forward_vector()
        wp = self.world.get_map().get_waypoint(loc)
        
        min_dist = float('inf')
        front_speed = 0
        front_vehicle = None
        
        for other in self.world.get_actors().filter('vehicle.*'):
            if other.id != vehicle.id:
                other_loc = other.get_location()
                other_wp = self.world.get_map().get_waypoint(other_loc)
                
                if other_wp.lane_id == wp.lane_id:
                    vec_to_other = other_loc - loc
                    forward_dot = vec_to_other.x * fwd.x + vec_to_other.y * fwd.y
                    
                    if forward_dot > 0:  # Vehicle is in front
                        dist = loc.distance(other_loc)
                        if dist < min_dist:
                            min_dist = dist
                            other_vel = other.get_velocity()
                            front_speed = math.sqrt(other_vel.x**2 + other_vel.y**2)
                            front_vehicle = other
        
        return {
            'distance': min_dist,
            'speed': front_speed,
            'vehicle': front_vehicle
        }

    def _attempt_overtake(self, vehicle, front_info, emergency=False):
        """Execute overtaking maneuver with enhanced decision making"""
        wp = self.world.get_map().get_waypoint(vehicle.get_location())
        left_lane = wp.get_left_lane()
        right_lane = wp.get_right_lane()
        
        # Calculate safety scores for each option
        left_score = self._calculate_lane_safety_score(vehicle, left_lane) if left_lane else -float('inf')
        right_score = self._calculate_lane_safety_score(vehicle, right_lane) if right_lane else -float('inf')
        
        # In emergency, lower the safety threshold
        safety_threshold = 0.3 if emergency else 0.7
        
        # Prefer left lane for overtaking if safe
        if left_score > safety_threshold and left_score >= right_score:
            self._execute_lane_change(vehicle, 'left', True)
        elif right_score > safety_threshold:
            self._execute_lane_change(vehicle, 'right', True)
        elif emergency and max(left_score, right_score) > 0:
            # In emergency, take the best available option
            self._execute_lane_change(vehicle, 'left' if left_score > right_score else 'right', True)

    def _calculate_lane_safety_score(self, vehicle, target_lane):
        """Calculate a safety score for a potential lane change"""
        if not target_lane:
            return -float('inf')
            
        loc = vehicle.get_location()
        score = 1.0
        
        for other in self.world.get_actors().filter('vehicle.*'):
            if other.id != vehicle.id:
                other_loc = other.get_location()
                other_wp = self.world.get_map().get_waypoint(other_loc)
                
                if other_wp.lane_id == target_lane.lane_id:
                    dist = loc.distance(other_loc)
                    
                    # Reduce score based on proximity of other vehicles
                    if dist < self.SAFE_DISTANCE:
                        score *= (dist / self.SAFE_DISTANCE)
                    
                    # Consider relative velocities
                    vel_diff = (vehicle.get_velocity().length() - 
                            other.get_velocity().length())
                    if vel_diff < 0:  # Other vehicle is faster
                        score *= 0.8
        
        return score

    def _check_lane_safe(self, vehicle, target_lane):
        """Check if target lane is safe for change"""
        if not target_lane:
            return False
            
        loc = vehicle.get_location()
        
        for other in self.world.get_actors().filter('vehicle.*'):
            if other.id != vehicle.id:
                other_loc = other.get_location()
                other_wp = self.world.get_map().get_waypoint(other_loc)
                
                if other_wp.lane_id == target_lane.lane_id:
                    dist = loc.distance(other_loc)
                    if dist < self.SAFE_DISTANCE:
                        return False
        
        return True

    def _execute_lane_change(self, vehicle, direction, is_overtaking=False):
        """Execute lane change with appropriate speed boost"""
        state = self.vehicles[vehicle.id]
        state['changing_lane'] = True
        state['change_start'] = time.time()
        state['boosted'] = True
        state['is_overtaking'] = is_overtaking
        
        # Apply speed boost
        boost_factor = self.OVERTAKE_BOOST if is_overtaking else self.SPEED_BOOST
        boosted_factor = state['normal_factor'] * boost_factor
        self.tm.vehicle_percentage_speed_difference(vehicle, boosted_factor)
        
        # Execute lane change
        self.tm.force_lane_change(vehicle, direction == 'left')

    def _emergency_lane_change(self, vehicle, waypoint):
        """Handle emergency lane change with higher tolerance"""
        left_lane = waypoint.get_left_lane()
        right_lane = waypoint.get_right_lane()
        
        # Try both lanes with reduced safety distance
        emergency_distance = self.SAFE_DISTANCE * 0.5
        if left_lane:
            if self._check_emergency_lane_safety(vehicle, left_lane, emergency_distance):
                self._execute_lane_change(vehicle, 'left', True)
                return
        if right_lane:
            if self._check_emergency_lane_safety(vehicle, right_lane, emergency_distance):
                self._execute_lane_change(vehicle, 'right', True)

    def _check_emergency_lane_safety(self, vehicle, target_lane, safety_distance):
        """Check lane safety with reduced safety distance"""
        loc = vehicle.get_location()
        
        for other in self.world.get_actors().filter('vehicle.*'):
            if other.id != vehicle.id:
                other_loc = other.get_location()
                other_wp = self.world.get_map().get_waypoint(other_loc)
                
                if other_wp.lane_id == target_lane.lane_id:
                    if loc.distance(other_loc) < safety_distance:
                        return False
        
        return True

def setup_world(client):
    """Setup CARLA world with Town04"""
    print("Loading Town04...")
    world = client.load_world('Town04')
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    spectator = world.get_spectator()
    spectator.set_transform(carla.Transform(
        carla.Location(x=0, y=25, z=70),
        carla.Rotation(pitch=-60)
    ))
    
    return world

def spawn_vehicles(world, highway_section, num_vehicles=30):
    """Spawn vehicles in highway section"""
    vehicles = []
    blueprint_library = world.get_blueprint_library()
    
    spawn_points = []
    map = world.get_map()
    waypoints = map.generate_waypoints(2.0)
    
    # Filter waypoints within highway section
    valid_waypoints = []
    for wp in waypoints:
        loc = wp.transform.location
        if (highway_section.x_min <= loc.x <= highway_section.x_max and 
            highway_section.y_min <= loc.y <= highway_section.y_max):
            valid_waypoints.append(wp)
    
    # Group by lanes
    lanes = {}
    for wp in valid_waypoints:
        if wp.lane_id not in lanes:
            lanes[wp.lane_id] = []
        lanes[wp.lane_id].append(wp)
    
    # Create spawn points for each lane
    for lane_waypoints in lanes.values():
        lane_waypoints.sort(key=lambda wp: wp.transform.location.x)
        step = max(1, len(lane_waypoints) // 8)
        for wp in lane_waypoints[::step]:
            transform = wp.transform
            transform.location.z += 0.5
            spawn_points.append(transform)
    
    # Filter vehicle blueprints
    vehicle_bps = [bp for bp in blueprint_library.filter('vehicle.*')
                  if bp.get_attribute('number_of_wheels').as_int() == 4]
    
    # Spawn vehicles
    for i in range(min(len(spawn_points), num_vehicles)):
        blueprint = random.choice(vehicle_bps)
        spawn_point = spawn_points[i]
        
        try:
            vehicle = world.try_spawn_actor(blueprint, spawn_point)
            if vehicle is not None:
                vehicles.append(vehicle)
                print(f"Spawned vehicle {vehicle.id}")
        except Exception as e:
            print(f"Failed to spawn vehicle: {e}")
    
    return vehicles

############################################################################################################################################################

def setup_csv_file():
    """Create CSV file with comprehensive data fields"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'training_data_{timestamp}.csv'
    
    headers = [
        # Vehicle Identification
        'Timestamp',
        'Vehicle_ID',
        
        # Current Vehicle State
        'Current_Lane',
        'Current_Speed',
        'Current_Acceleration',
        'Distance_In_Current_Lane',
        
        # Front Vehicle Information
        'Front_Vehicle_Distance',
        'Front_Vehicle_Speed',
        'Time_To_Collision_Front',
        'Front_Vehicle_Lane',
        
        # Left Lane Information
        'Left_Front_Vehicle_Distance',
        'Left_Front_Vehicle_Speed',
        'Left_Rear_Vehicle_Distance',
        'Left_Rear_Vehicle_Speed',
        
        # Right Lane Information
        'Right_Front_Vehicle_Distance',
        'Right_Front_Vehicle_Speed',
        'Right_Rear_Vehicle_Distance',
        'Right_Rear_Vehicle_Speed',
        
        # Lane Change Information
        'Time_Since_Last_Change',
        'Last_Change_Direction',
        'Current_Lane_Change_State',
        'Lane_Change_Success',
        
        # Safety Metrics
        'Min_Safety_Distance',
        'Near_Miss_Incident',
        'Safety_Margin',
        'Collision_Occurred',
        
        # Performance Metrics
        'Average_Speed_Last_10s',
        'Successful_Overtakes',
        'Time_Following_Slower_Vehicle',
        'Lane_Change_Duration',
        
        # Action Taken
        'Action_Taken',  # none, left_change, right_change, speed_adjust
        'Action_Success'
    ]
    
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    
    return filename

def collect_vehicle_data(traffic_manager, vehicle, world):
    """Collect comprehensive data for a single vehicle"""
    if not vehicle.is_alive:
        return None
        
    try:
        # Get basic vehicle information
        loc = vehicle.get_location()
        vel = vehicle.get_velocity()
        acc = vehicle.get_acceleration()
        current_speed = 3.6 * math.sqrt(vel.x**2 + vel.y**2)  # km/h
        current_acc = math.sqrt(acc.x**2 + acc.y**2)
        
        # Get waypoint information
        waypoint = world.get_map().get_waypoint(loc)
        current_lane = waypoint.lane_id
        
        # Get vehicle state from traffic manager
        state = traffic_manager.vehicles.get(vehicle.id, {})
        
        # Get surrounding vehicles information
        front_info = get_surrounding_vehicle_info(vehicle, world, 'front')
        left_info = get_surrounding_vehicle_info(vehicle, world, 'left')
        right_info = get_surrounding_vehicle_info(vehicle, world, 'right')
        
        # Calculate safety metrics
        safety_metrics = calculate_safety_metrics(vehicle, front_info, left_info, right_info)
        
        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(state)
        
        # Prepare data row
        data = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
            vehicle.id,
            current_lane,
            round(current_speed, 2),
            round(current_acc, 2),
            state.get('distance_in_lane', 0),
            
            # Front vehicle data
            round(front_info['distance'], 2),
            round(front_info['speed'], 2),
            round(front_info['ttc'], 2),
            front_info['lane'],
            
            # Left lane data
            round(left_info['front_distance'], 2),
            round(left_info['front_speed'], 2),
            round(left_info['rear_distance'], 2),
            round(left_info['rear_speed'], 2),
            
            # Right lane data
            round(right_info['front_distance'], 2),
            round(right_info['front_speed'], 2),
            round(right_info['rear_distance'], 2),
            round(right_info['rear_speed'], 2),
            
            # Lane change data
            round(time.time() - state.get('last_lane_change_time', 0), 2),
            state.get('last_lane_change_direction', 'none'),
            state.get('changing_lane', False),
            state.get('last_change_success', True),
            
            # Safety metrics
            round(safety_metrics['min_safety_distance'], 2),
            safety_metrics['near_miss'],
            round(safety_metrics['safety_margin'], 2),
            safety_metrics['collision_occurred'],
            
            # Performance metrics
            round(performance_metrics['avg_speed'], 2),
            performance_metrics['successful_overtakes'],
            round(performance_metrics['time_following'], 2),
            round(performance_metrics['lane_change_duration'], 2),
            
            # Action information
            state.get('current_action', 'none'),
            state.get('action_success', True)
        ]
        
        return data
        
    except Exception as e:
        print(f"Error collecting data for vehicle {vehicle.id}: {e}")
        return None

def get_surrounding_vehicle_info(vehicle, world, direction):
    """Get information about surrounding vehicles with proper error handling"""
    loc = vehicle.get_location()
    fwd = vehicle.get_transform().get_forward_vector()
    wp = world.get_map().get_waypoint(loc)  # Use world map to get waypoint
    
    # Initialize return structure with default values
    info = {
        'distance': float('inf'),
        'speed': 0,
        'ttc': float('inf'),
        'lane': None,
        'front_distance': float('inf'),
        'front_speed': 0,
        'rear_distance': float('inf'),
        'rear_speed': 0
    }
    
    try:
        for other in world.get_actors().filter('vehicle.*'):
            if other.id != vehicle.id and other.is_alive:
                other_loc = other.get_location()
                other_wp = world.get_map().get_waypoint(other_loc)  # Use world map
                dist = loc.distance(other_loc)
                
                if direction == 'front' and other_wp.lane_id == wp.lane_id:
                    vec_to_other = other_loc - loc
                    if vec_to_other.dot(fwd) > 0:  # Vehicle is in front
                        if dist < info['distance']:
                            info['distance'] = dist
                            other_vel = other.get_velocity()
                            info['speed'] = 3.6 * math.sqrt(other_vel.x**2 + other_vel.y**2)
                            info['ttc'] = calculate_ttc(vehicle, other)
                            info['lane'] = other_wp.lane_id
                            
                elif direction in ['left', 'right']:
                    target_lane = wp.get_left_lane() if direction == 'left' else wp.get_right_lane()
                    if target_lane and other_wp.lane_id == target_lane.lane_id:
                        vec_to_other = other_loc - loc
                        if vec_to_other.dot(fwd) > 0:  # Vehicle is in front
                            if dist < info['front_distance']:
                                info['front_distance'] = dist
                                other_vel = other.get_velocity()
                                info['front_speed'] = 3.6 * math.sqrt(other_vel.x**2 + other_vel.y**2)
                        else:  # Vehicle is behind
                            if dist < info['rear_distance']:
                                info['rear_distance'] = dist
                                other_vel = other.get_velocity()
                                info['rear_speed'] = 3.6 * math.sqrt(other_vel.x**2 + other_vel.y**2)
    except Exception as e:
        print(f"Error getting surrounding vehicle info: {str(e)}")
    
    return info

def calculate_ttc(ego_vehicle, other_vehicle):
    """Calculate Time To Collision"""
    ego_vel = ego_vehicle.get_velocity()
    other_vel = other_vehicle.get_velocity()
    
    relative_velocity = math.sqrt(
        (ego_vel.x - other_vel.x)**2 +
        (ego_vel.y - other_vel.y)**2
    )
    
    if relative_velocity > 0.1:  # Avoid division by near-zero
        distance = ego_vehicle.get_location().distance(other_vehicle.get_location())
        return distance / relative_velocity
    return float('inf')

def calculate_safety_metrics(vehicle, front_info, left_info, right_info):
    """Calculate safety-related metrics"""
    return {
        'min_safety_distance': min(
            front_info['distance'],
            left_info['front_distance'],
            right_info['front_distance']
        ),
        'near_miss': any(d < 5.0 for d in [
            front_info['distance'],
            left_info['front_distance'],
            right_info['front_distance']
        ]),
        'safety_margin': calculate_safety_margin(front_info, left_info, right_info),
        'collision_occurred': False  # Updated by collision sensor
    }

def calculate_performance_metrics(state):
    """Calculate performance-related metrics"""
    return {
        'avg_speed': np.mean(state.get('speed_history', [0])),
        'successful_overtakes': state.get('successful_overtakes', 0),
        'time_following': state.get('time_following', 0),
        'lane_change_duration': state.get('last_change_duration', 0)
    }

def calculate_safety_margin(front_info, left_info, right_info):
    """Calculate overall safety margin with proper error handling"""
    distances = []
    
    # Only add finite distances to the list
    if front_info['distance'] != float('inf'):
        distances.append(front_info['distance'])
    if left_info['front_distance'] != float('inf'):
        distances.append(left_info['front_distance'])
    if left_info['rear_distance'] != float('inf'):
        distances.append(left_info['rear_distance'])
    if right_info['front_distance'] != float('inf'):
        distances.append(right_info['front_distance'])
    if right_info['rear_distance'] != float('inf'):
        distances.append(right_info['rear_distance'])
    
    # Return appropriate value based on whether we have any valid distances
    return min(distances) if distances else float('inf')


############################################################################################################################################################
def main():
    vehicles = []
    traffic_manager = None
    world = None
    csv_file = None

    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # Setup world and highway section
        print("Loading Town04...")
        world = client.load_world('Town04')
        highway = HighwaySection()
        
        # Setup world settings
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Setup spectator
        spectator = world.get_spectator()
        spectator.set_transform(carla.Transform(
            carla.Location(x=0, y=25, z=70),
            carla.Rotation(pitch=-60)
        ))
        
        # Create traffic manager and spawn vehicles
        traffic_manager = SimpleTrafficManager(client, world)
        vehicles = spawn_vehicles(world, highway, num_vehicles=30)
        
        if not vehicles:
            raise RuntimeError("No vehicles spawned!")
        
        print(f"Successfully spawned {len(vehicles)} vehicles")
        
        # Initialize vehicles
        for vehicle in vehicles:
            traffic_manager.setup_vehicle(vehicle)
        
        # Setup data collection
        csv_filename = setup_csv_file()
        print(f"Logging data to: {csv_filename}")
        
        # Main simulation loop
        with open(csv_filename, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            while True:
                start_time = time.time()
                
                # Update and collect data for each vehicle
                for vehicle in vehicles:
                    if not vehicle.is_alive:
                        continue
                    
                    try:
                        # Update vehicle behavior
                        traffic_manager.update_vehicle(vehicle)
                        
                        # Check if vehicle is in bounds
                        loc = vehicle.get_location()
                        if not (highway.x_min <= loc.x <= highway.x_max and 
                               highway.y_min <= loc.y <= highway.y_max):
                            spawn_points = traffic_manager._get_spawn_points()
                            if spawn_points:
                                vehicle.set_transform(random.choice(spawn_points))
                                print(f"Respawned vehicle {vehicle.id} (out of bounds)")
                        
                        # Collect and write data
                        data = collect_vehicle_data(traffic_manager, vehicle, world)
                        if data:
                            writer.writerow(data)
                            csv_file.flush()  # Ensure data is written
                    
                    except Exception as e:
                        print(f"Error processing vehicle {vehicle.id}: {e}")
                        continue
                
                # Tick world and maintain frame rate
                world.tick()
                
                # Maintain fixed time step (20 FPS)
                elapsed = time.time() - start_time
                if elapsed < 0.05:
                    time.sleep(0.05 - elapsed)
                
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\nCleaning up...")
        try:
            # Close CSV file if open
            if csv_file and not csv_file.closed:
                csv_file.close()
            
            # Cleanup collision sensors
            if traffic_manager:
                for sensor in traffic_manager.collision_sensors.values():
                    if sensor.is_alive:
                        sensor.destroy()
            
            # Cleanup vehicles
            for vehicle in vehicles:
                try:
                    if vehicle.is_alive:
                        vehicle.destroy()
                except:
                    pass
            
            # Reset world settings
            if world:
                settings = world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.apply_settings(settings)
            
            print("Cleanup completed successfully")
            
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")

if __name__ == '__main__':
    main()