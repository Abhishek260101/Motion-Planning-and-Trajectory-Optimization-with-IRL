import carla
import numpy as np
import math
import csv
import time
import cv2
import os
import random
from datetime import datetime
import queue
import threading
from tensorflow.keras.models import load_model

class HybridController:
    def __init__(self):
        # Constants
        self.WIDTH = 800
        self.HEIGHT = 600
        self.SEMANTIC_WIDTH = 320
        self.SEMANTIC_HEIGHT = 240
        self.FOV = 90
        self.CAMERA_POS_Z = 1.3
        self.CAMERA_POS_X = 1.4
        
        self.csv_file = None
        self.csv_writer = None

        # Navigation parameters
        self.MIN_DISTANCE_TO_TARGET = 5.0  # meters
        self.MIN_SPAWN_DISTANCE = 100.0    # meters
        self.current_destination = None
        
        # Control parameters
        self.PID_params = {
            'Kp': 1.0,
            'Ki': 0.1,
            'Kd': 0.1,
            'dt': 0.05
        }
        
        # Initialize CARLA
        print("Connecting to CARLA...")
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Vehicle and sensor initialization
        self.vehicle = None
        self.rgb_camera = None
        self.sem_camera = None
        self.collision_sensor = None
        
        # Sensor data storage
        self.sensor_data = {
            'rgb': None,
            'semantic': None,
            'collision': []
        }
        
        # Load CNN model
        print("Loading CNN model...")
        self.cnn_model = load_model('model_saved_from_CNN.h5', compile=False)
        self.cnn_model.compile()
        
        # Setup data collection
        self.setup_data_collection()
        
        # Control state
        self.prev_error = 0
        self.integral = 0
        self.target_speed = 30  # km/h
        


    def setup_data_collection(self):
        """Setup directories and files for data collection"""
        try:
            self.data_dir = f"hybrid_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(f"{self.data_dir}/rgb_images", exist_ok=True)
            os.makedirs(f"{self.data_dir}/semantic_images", exist_ok=True)
            
            # Keep CSV file open during data collection
            self.csv_file = open(f"{self.data_dir}/driving_data.csv", 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'timestamp', 'pos_x', 'pos_y', 'pos_z',
                'velocity', 'steering', 'throttle', 'brake',
                'waypoint_x', 'waypoint_y', 'distance_to_target',
                'turn_direction'
            ])
            print("Data collection setup complete")
        except Exception as e:
            print(f"Error in setup_data_collection: {e}")

    def setup_sensors(self):
        """Setup all required sensors"""
        # RGB Camera
        rgb_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(self.WIDTH))
        rgb_bp.set_attribute('image_size_y', str(self.HEIGHT))
        rgb_bp.set_attribute('fov', str(self.FOV))
        
        # Semantic Segmentation Camera
        sem_bp = self.world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        sem_bp.set_attribute('image_size_x', str(self.SEMANTIC_WIDTH))
        sem_bp.set_attribute('image_size_y', str(self.SEMANTIC_HEIGHT))
        sem_bp.set_attribute('fov', str(self.FOV))
        
        # Transform for cameras
        camera_transform = carla.Transform(
            carla.Location(x=self.CAMERA_POS_X, z=self.CAMERA_POS_Z)
        )
        
        # Spawn cameras
        self.rgb_camera = self.world.spawn_actor(rgb_bp, camera_transform, attach_to=self.vehicle)
        self.sem_camera = self.world.spawn_actor(sem_bp, camera_transform, attach_to=self.vehicle)
        
        # Setup callbacks
        self.rgb_camera.listen(lambda image: self._process_rgb_image(image))
        self.sem_camera.listen(lambda image: self._process_semantic_image(image))
        
        # Collision sensor
        col_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(col_bp, camera_transform, attach_to=self.vehicle)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def _process_rgb_image(self, image):
        """Process RGB camera data"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.sensor_data['rgb'] = array

    def _process_semantic_image(self, image):
        """Process semantic segmentation data"""
        image.convert(carla.ColorConverter.CityScapesPalette)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
        self.sensor_data['semantic'] = array

    def _on_collision(self, event):
        """Handle collision events"""
        self.sensor_data['collision'].append(event)

    def get_lane_info(self):
        """Get lane information from semantic segmentation"""
        if self.sensor_data['semantic'] is None:
            return None, None
        
        # Process semantic image with CNN model
        semantic_processed = self._preprocess_semantic(self.sensor_data['semantic'])
        lane_deviation = self._analyze_lane_position(semantic_processed)
        
        return semantic_processed, lane_deviation

    def _preprocess_semantic(self, semantic_image):
        """Preprocess semantic image for CNN"""
        try:
            # Crop image first (matching original environment.py logic)
            height_from = int(self.SEMANTIC_HEIGHT * (1 - 0.5))  # Using HEIGHT_REQUIRED_PORTION = 0.5
            width_from = int((self.SEMANTIC_WIDTH - self.SEMANTIC_WIDTH * 0.9) / 2)  # WIDTH_REQUIRED_PORTION = 0.9
            width_to = width_from + int(self.SEMANTIC_WIDTH * 0.9)
            
            cropped_img = semantic_image[height_from:, width_from:width_to]
            
            # Resize to match model's expected input
            resized_img = cv2.resize(cropped_img, (288, 120))
            
            # Convert to float and normalize
            img = np.float32(resized_img)
            img = img / 255
            img = np.expand_dims(img, axis=0)
            
            # Add second input (0) as required by the model
            cnn_applied = self.cnn_model([img, 0], training=False)
            return cnn_applied
        except Exception as e:
            print(f"Error in preprocessing semantic image: {e}")
            return None

    def _analyze_lane_position(self, processed_image):
        """Analyze lane position from processed semantic image"""
        # Your lane analysis logic here
        # Return lane deviation (negative: left, positive: right)
        return 0.0  # Placeholder

    def get_waypoint_control(self):
        """Get control based on waypoints with fixed steering logic"""
        if not self.vehicle:
            return 0, 0, 0
        
        # Get current waypoint and future waypoints
        current_loc = self.vehicle.get_location()
        current_wp = self.world.get_map().get_waypoint(current_loc, project_to_road=True)
        
        if not current_wp:
            return 0, 0, 0
            
        # Get next waypoint along the road
        next_wps = current_wp.next(2.0)  # Get waypoint 2 meters ahead
        if not next_wps:
            return 0, 0, 0
            
        next_wp = next_wps[0]
        
        # Calculate steering based on waypoint direction
        current_transform = self.vehicle.get_transform()
        
        # Calculate angle between vehicle's forward vector and next waypoint
        wp_vector = next_wp.transform.location - current_loc
        wp_vector_norm = math.sqrt(wp_vector.x**2 + wp_vector.y**2)
        
        if wp_vector_norm < 0.001:
            return 0, 0, 0
        
        # Normalize waypoint vector
        wp_vector.x /= wp_vector_norm
        wp_vector.y /= wp_vector_norm
        
        # Get vehicle's forward vector
        fwd = current_transform.get_forward_vector()
        
        # Calculate angle
        angle = math.atan2(wp_vector.y, wp_vector.x) - math.atan2(fwd.y, fwd.x)
        
        # Normalize angle to [-pi, pi]
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        
        # Calculate steering (with reduced sensitivity)
        steering = max(-1.0, min(1.0, angle * 0.5))  # Reduced from 1.0 to 0.5
        
        # Speed control
        current_speed = self._get_speed()
        if current_speed < self.target_speed - 5:
            throttle = 0.7
            brake = 0.0
        elif current_speed > self.target_speed + 5:
            throttle = 0.0
            brake = 0.3
        else:
            throttle = 0.5
            brake = 0.0
            
        return steering, throttle, brake

    def _calculate_control(self, waypoints):
        """Calculate control values based on waypoints"""
        if not waypoints:
            return 0, 0, 0
        
        # Get vehicle state
        vehicle_transform = self.vehicle.get_transform()
        vehicle_loc = vehicle_transform.location
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
        
        # Calculate path angle
        target_wp = waypoints[min(5, len(waypoints)-1)]
        target_loc = target_wp.transform.location
        
        # Calculate angle to target
        target_angle = math.atan2(
            target_loc.y - vehicle_loc.y,
            target_loc.x - vehicle_loc.x
        )
        
        # Calculate steering
        angle_diff = target_angle - vehicle_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        steering = max(-1.0, min(1.0, angle_diff))
        
        # Calculate speed and throttle
        current_speed = self._get_speed()
        if current_speed < self.target_speed - 5:
            throttle = 0.7
            brake = 0.0
        elif current_speed > self.target_speed + 5:
            throttle = 0.0
            brake = 0.3
        else:
            throttle = 0.5
            brake = 0.0
            
        return steering, throttle, brake

    def _get_speed(self):
        """Get current speed in km/h"""
        vel = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
    
    def get_next_waypoint(self):
        """Get next waypoint with random turn decisions"""
        if not self.vehicle:
            return None
            
        # Get current waypoint
        current_loc = self.vehicle.get_location()
        current_wp = self.world.get_map().get_waypoint(current_loc)
        
        if not current_wp:
            return None
        
        # Randomly decide to make a turn when possible
        if random.random() < 0.2:  # 20% chance to consider a turn
            # Get possible next waypoints including turns
            next_possible_wps = []
            
            # Try to get left turns
            left_wps = current_wp.get_left_lane()
            if left_wps and left_wps.lane_type == carla.LaneType.Driving:
                next_possible_wps.extend(left_wps.next(20.0))
                
            # Try to get right turns
            right_wps = current_wp.get_right_lane()
            if right_wps and right_wps.lane_type == carla.LaneType.Driving:
                next_possible_wps.extend(right_wps.next(20.0))
                
            # Add straight option
            next_possible_wps.extend(current_wp.next(20.0))
            
            if next_possible_wps:
                return random.choice(next_possible_wps)
        
        # Default to straight if no turn or no turn possible
        next_wps = current_wp.next(2.0)
        return next_wps[0] if next_wps else None

    def control_step(self):
        """Execute one control step with random navigation"""
        try:
            # Get next waypoint with possible turn
            next_wp = self.get_next_waypoint()
            if not next_wp:
                return None
                
            # Calculate controls
            current_transform = self.vehicle.get_transform()
            target_loc = next_wp.transform.location
            
            # Calculate steering angle
            direction = target_loc - current_transform.location
            direction_norm = math.sqrt(direction.x**2 + direction.y**2)
            
            if direction_norm > 0.001:
                # Calculate desired steering
                forward = current_transform.get_forward_vector()
                dot = forward.x * direction.x + forward.y * direction.y
                cross = forward.x * direction.y - forward.y * direction.x
                steer_angle = math.atan2(cross, dot)
                
                # Apply steering with smoothing
                steer = max(-1.0, min(1.0, steer_angle))
            else:
                steer = 0.0
                
            # Speed control
            current_speed = self._get_speed()
            if current_speed < self.target_speed - 5:
                throttle = 0.7
                brake = 0.0
            elif current_speed > self.target_speed + 5:
                throttle = 0.0
                brake = 0.3
            else:
                throttle = 0.5
                brake = 0.0
                
            # Create control command
            control = carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=brake
            )
            
            # Save data point
            self.save_data_point(control)
            
            return control
            
        except Exception as e:
            print(f"Error in control_step: {e}")
            return None

    def _get_distance_to_target(self):
        """Calculate distance to current destination"""
        if not self.vehicle or not self.current_destination:
            return 0.0
            
        current_location = self.vehicle.get_location()
        return current_location.distance(self.current_destination.location)

    def save_data_point(self, control):
        """Save data point with current state"""
        try:
            if self.csv_writer is None:
                return
                
            # Get current state
            transform = self.vehicle.get_transform()
            velocity = self.vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Get current waypoint
            current_wp = self.world.get_map().get_waypoint(transform.location)
            
            # Get distance to target safely
            try:
                distance_to_target = self._get_distance_to_target()
            except Exception:
                distance_to_target = 0.0
            
            # Write data
            self.csv_writer.writerow([
                time.time(),
                transform.location.x,
                transform.location.y,
                transform.location.z,
                speed,
                control.steer,
                control.throttle,
                control.brake,
                current_wp.transform.location.x if current_wp else 0,
                current_wp.transform.location.y if current_wp else 0,
                distance_to_target,
                'straight' if abs(control.steer) < 0.1 else ('left' if control.steer < 0 else 'right')
            ])
            self.csv_file.flush()  # Ensure data is written
            
        except Exception as e:
            print(f"Error saving data point: {e}")

    def _calculate_cross_track_error(self, waypoint):
        """Calculate cross track error"""
        vehicle_loc = self.vehicle.get_transform().location
        wp_loc = waypoint.transform.location
        return math.sqrt((wp_loc.x - vehicle_loc.x)**2 + 
                        (wp_loc.y - vehicle_loc.y)**2)

    def _calculate_curvature(self, waypoint):
        """Calculate road curvature"""
        next_wps = waypoint.next(2.0)
        if not next_wps:
            return 0
            
        wp1 = waypoint.transform.location
        wp2 = next_wps[0].transform.location
        
        dx = wp2.x - wp1.x
        dy = wp2.y - wp1.y
        
        if dx == 0:
            return float('inf')
            
        curvature = abs(dy / (dx * dx))
        return curvature

    def get_random_destination(self):
        """Get a random destination far from current position"""
        spawn_points = self.world.get_map().get_spawn_points()
        
        if self.vehicle:
            current_location = self.vehicle.get_location()
            far_points = [p for p in spawn_points if 
                         current_location.distance(p.location) > self.MIN_SPAWN_DISTANCE]
            if far_points:
                return random.choice(far_points)
        
        return random.choice(spawn_points)
    
    def check_destination_reached(self):
        """Check if current destination is reached"""
        if not self.current_destination or not self.vehicle:
            return False
            
        current_location = self.vehicle.get_location()
        return current_location.distance(
            self.current_destination.location) < self.MIN_DISTANCE_TO_TARGET
    
    def collect_data(self, duration=3600):
        """Collect driving data with improved navigation"""
        print(f"Starting data collection for {duration} seconds...")
        try:
            self.reset_vehicle()  # Initial spawn
            start_time = time.time()
            
            while time.time() - start_time < duration:
                try:
                    if len(self.sensor_data['collision']) > 0:
                        print("Collision detected! Respawning...")
                        self.reset_vehicle()
                        continue
                    
                    # Check if destination reached
                    if self.check_destination_reached():
                        print("Destination reached! Setting new destination...")
                        self.current_destination = self.get_random_destination()
                        print(f"New destination set at: {self.current_destination.location}")
                    
                    control = self.control_step()
                    if control is not None:
                        self.vehicle.apply_control(control)
                        self.save_data_point(control)
                    
                    time.sleep(self.PID_params['dt'])
                    
                except KeyboardInterrupt:
                    print("\nStopping data collection...")
                    break
                except Exception as e:
                    print(f"Error during control step: {e}")
                    
        except Exception as e:
            print(f"Error in data collection: {e}")
        finally:
            print("Cleaning up...")
            self.cleanup()

    def reopen_csv_file(self):
        """Reopen CSV file if closed"""
        try:
            if self.csv_file is None or self.csv_file.closed:
                self.csv_file = open(f"{self.data_dir}/driving_data.csv", 'a', newline='')  # 'a' for append
                self.csv_writer = csv.writer(self.csv_file)
        except Exception as e:
            print(f"Error reopening CSV file: {e}")

    def cleanup_actors(self):
        """Cleanup only CARLA actors"""
        try:
            actors_to_clean = []
            if hasattr(self, 'rgb_camera') and self.rgb_camera is not None:
                actors_to_clean.append(self.rgb_camera)
            if hasattr(self, 'sem_camera') and self.sem_camera is not None:
                actors_to_clean.append(self.sem_camera)
            if hasattr(self, 'collision_sensor') and self.collision_sensor is not None:
                actors_to_clean.append(self.collision_sensor)
            if hasattr(self, 'vehicle') and self.vehicle is not None:
                actors_to_clean.append(self.vehicle)
                
            for actor in actors_to_clean:
                try:
                    actor.destroy()
                except Exception as e:
                    print(f"Error destroying actor: {e}")
        except Exception as e:
            print(f"Error during actor cleanup: {e}")
            
    def reset_vehicle(self):
        """Reset vehicle after collision"""
        self.cleanup_actors()  # Only clean up CARLA actors
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = random.choice(spawn_points)
        
        blueprint = self.world.get_blueprint_library().filter('model3')[0]
        self.vehicle = self.world.spawn_actor(blueprint, spawn_point)
        
        self.setup_sensors()
        self.sensor_data['collision'] = []
        self.reopen_csv_file()  # Reopen CSV file
        time.sleep(2)

    def cleanup(self):
        """Final cleanup of everything"""
        try:
            self.cleanup_actors()
        finally:
            if hasattr(self, 'csv_file') and not self.csv_file.closed:
                self.csv_file.close()
                print("CSV file closed successfully")
def main():
    controller = None
    try:
        controller = HybridController()
        print("Starting data collection...")
        controller.collect_data(duration=10800)  # 3 hour
    except KeyboardInterrupt:
        print("\nStopping data collection...")
    except Exception as e:
        print(f"Error in main: {e}")
    finally:
        if controller:
            controller.cleanup()

if __name__ == "__main__":
    main()
