'''
Modified RL environment maintaining original functionality with trajectory features
'''

import random
import time
import numpy as np
import math 
import cv2
import gym
from gym import spaces
import carla
from tensorflow.keras.models import load_model

SECONDS_PER_EPISODE = 15
N_CHANNELS = 3
HEIGHT = 240
WIDTH = 320
SPIN = 10

HEIGHT_REQUIRED_PORTION = 0.5
WIDTH_REQUIRED_PORTION = 0.9

SHOW_PREVIEW = True

TRAJECTORY_TIME_HORIZON = 3.0  # seconds to look ahead
TRAJECTORY_TIMESTEP = 0.2      # timestep for trajectory points
NUM_TRAJECTORIES = 20          # number of trajectories to generate
SAFETY_DISTANCE = 3.0          # meters
MIN_TURNING_RADIUS = 6.0       # meters

class CarEnv(gym.Env):
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = WIDTH
    im_height = HEIGHT
    front_camera = None
    CAMERA_POS_Z = 1.3 
    CAMERA_POS_X = 1.4
    PREFERRED_SPEED = 30
    SPEED_THRESHOLD = 2

    def __init__(self):
        super(CarEnv, self).__init__()
        
        # Keep original action space
        self.action_space = spaces.MultiDiscrete([9])
        
        self.height_from = int(HEIGHT * (1 - HEIGHT_REQUIRED_PORTION))
        self.width_from = int((WIDTH - WIDTH * WIDTH_REQUIRED_PORTION) / 2)
        self.width_to = self.width_from + int(WIDTH_REQUIRED_PORTION * WIDTH)
        self.new_height = HEIGHT - self.height_from
        self.new_width = self.width_to - self.width_from
        self.image_for_CNN = None
        
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(7, 18, 8), dtype=np.float32)
        
        # CARLA setup
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = not self.SHOW_CAM
        self.world.apply_settings(self.settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.cnn_model = load_model('C:\SelfDrive\RL_Full_Tutorial\model_saved_from_CNN.h5', compile=False)
        self.cnn_model.compile()
        
        if self.SHOW_CAM:
            self.spectator = self.world.get_spectator()

    def maintain_speed(self, s):
        if s >= self.PREFERRED_SPEED:
            return 0
        elif s < self.PREFERRED_SPEED - self.SPEED_THRESHOLD:
            return 0.7
        else:
            return 0.3

    def generate_trajectories(self):
        """Generate multiple possible trajectories"""
        trajectories = []
        current_transform = self.vehicle.get_transform()
        current_velocity = self.vehicle.get_velocity()
        current_speed = math.sqrt(current_velocity.x**2 + current_velocity.y**2)
        
        # Get current waypoint for route following
        current_waypoint = self.world.get_map().get_waypoint(current_transform.location)
        
        # Generate acceleration and steering combinations
        accelerations = [-2.0, -1.0, 0.0, 1.0, 2.0]  # m/s²
        steerings = [-0.3, -0.15, 0, 0.15, 0.3]      # radians
        
        for acc in accelerations:
            for steer in steerings:
                trajectory = []
                # Start from current position
                x, y = 0, 0  # Local coordinates
                yaw = 0      # Local heading
                speed = current_speed
                
                # Project future positions
                for t in np.arange(0, TRAJECTORY_TIME_HORIZON, TRAJECTORY_TIMESTEP):
                    # Update speed with acceleration
                    speed += acc * TRAJECTORY_TIMESTEP
                    speed = max(0, min(speed, self.PREFERRED_SPEED))  # Bound speed
                    
                    # Calculate turning radius and angular velocity
                    if abs(steer) > 0.01:
                        turn_radius = MIN_TURNING_RADIUS / steer
                        angular_velocity = speed / turn_radius
                    else:
                        angular_velocity = 0
                    
                    # Update position and heading
                    yaw += angular_velocity * TRAJECTORY_TIMESTEP
                    x += speed * math.cos(yaw) * TRAJECTORY_TIMESTEP
                    y += speed * math.sin(yaw) * TRAJECTORY_TIMESTEP
                    
                    # Transform to world coordinates
                    world_x = x * math.cos(math.radians(current_transform.rotation.yaw)) - \
                             y * math.sin(math.radians(current_transform.rotation.yaw)) + \
                             current_transform.location.x
                    world_y = x * math.sin(math.radians(current_transform.rotation.yaw)) + \
                             y * math.cos(math.radians(current_transform.rotation.yaw)) + \
                             current_transform.location.y
                    world_yaw = yaw + math.radians(current_transform.rotation.yaw)
                    
                    trajectory.append({
                        'location': carla.Location(x=world_x, y=world_y, z=current_transform.location.z),
                        'yaw': world_yaw,
                        'speed': speed,
                        'acceleration': acc,
                        'steering': steer
                    })
                
                trajectories.append(trajectory)
        
        return trajectories

    def check_trajectory_safety(self, trajectory):
        """Check if a trajectory is safe"""
        try:
            for point in trajectory:
                # Check collision with other vehicles
                for actor in self.world.get_actors().filter('vehicle.*'):
                    if actor.id != self.vehicle.id:
                        if actor.get_location().distance(point['location']) < SAFETY_DISTANCE:
                            return False
                
                # Check if point is on road
                waypoint = self.world.get_map().get_waypoint(point['location'])
                if not waypoint or waypoint.lane_type != carla.LaneType.Driving:
                    return False
                
                # Check speed limits
                if point['speed'] > self.PREFERRED_SPEED * 1.1:  # 10% tolerance
                    return False
                
                # Check acceleration limits
                if abs(point['acceleration']) > 3.0:  # m/s²
                    return False
                
            return True
        except Exception as e:
            print(f"Safety check error: {e}")
            return False

    def select_best_trajectory(self, safe_trajectories):
        """Select best trajectory based on progress and safety"""
        if not safe_trajectories:
            return None
            
        best_score = float('-inf')
        best_trajectory = None
        
        for trajectory in safe_trajectories:
            score = 0
            
            # Progress along route
            end_point = trajectory[-1]['location']
            distance = self.initial_location.distance(end_point)
            score += distance * 2.0
            
            # Penalize deviation from preferred speed
            speed_diff = abs(trajectory[-1]['speed'] - self.PREFERRED_SPEED)
            score -= speed_diff
            
            # Penalize large steering angles
            score -= abs(trajectory[0]['steering']) * 5.0
            
            if score > best_score:
                best_score = score
                best_trajectory = trajectory
        
        return best_trajectory
    
    def step(self, action):
        # Original camera and spectator handling
        trans = self.vehicle.get_transform()
        if self.SHOW_CAM:
            self.spectator.set_transform(carla.Transform(
                trans.location + carla.Location(z=20),
                carla.Rotation(yaw=-180, pitch=-90)
            ))

        self.step_counter += 1

        # Generate and filter trajectories
        trajectories = self.generate_trajectories()
        safe_trajectories = [t for t in trajectories if self.check_trajectory_safety(t)]
        
        # Select trajectory based on action
        if safe_trajectories:
            trajectory_idx = action[0] % len(safe_trajectories)
            selected_trajectory = safe_trajectories[trajectory_idx]
        else:
            # Emergency fallback
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, brake=1.0))
            return self.image_for_CNN, -100, True, {}
        
        # Apply first control from trajectory
        first_point = selected_trajectory[1]  # Use second point as target
        current_transform = self.vehicle.get_transform()
        
        # Calculate control inputs
        target_speed = first_point['speed']
        current_velocity = self.vehicle.get_velocity()
        current_speed = math.sqrt(current_velocity.x**2 + current_velocity.y**2)
        
        # Speed control using your original maintain_speed function
        kmh = int(current_speed * 3.6)  # Convert to kmh
        estimated_throttle = self.maintain_speed(kmh)
        
        # Steering control
        target_yaw = first_point['yaw']
        current_yaw = math.radians(current_transform.rotation.yaw)
        yaw_diff = math.atan2(math.sin(target_yaw - current_yaw), 
                            math.cos(target_yaw - current_yaw))
        steer = max(-1.0, min(1.0, yaw_diff))

        # Apply control
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=estimated_throttle,
            steer=steer,
            brake=0.0
        ))

        # Rest of your original step function remains the same
        distance_travelled = self.initial_location.distance(self.vehicle.get_location())

        # Camera handling
        cam = self.front_camera
        if self.SHOW_CAM:
            cv2.imshow('Sem Camera', cam)
            cv2.waitKey(1)

        # Steering lock logic
        lock_duration = 0
        if self.steering_lock == False:
            if steer < -0.6 or steer > 0.6:
                self.steering_lock = True
                self.steering_lock_start = time.time()
        else:
            if steer < -0.6 or steer > 0.6:
                lock_duration = time.time() - self.steering_lock_start

        # Original reward calculation
        reward = 0
        done = False
        
        if len(self.collision_hist) != 0:
            done = True
            reward = reward - 300
            self.cleanup()
        if len(self.lane_invade_hist) != 0:
            done = True
            reward = reward - 300
            self.cleanup()
            
        if lock_duration > 3:
            reward = reward - 150
            done = True
            self.cleanup()
        elif lock_duration > 1:
            reward = reward - 20

        if distance_travelled < 30:
            reward = reward - 1
        elif distance_travelled < 50:
            reward = reward + 1
        else:
            reward = reward + 2

        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True
            self.cleanup()

        self.image_for_CNN = self.apply_cnn(self.front_camera[self.height_from:, self.width_from:self.width_to])

        return self.image_for_CNN, reward, done, {}

    def reset(self):
        self.collision_hist = []
        self.lane_invade_hist = []
        self.actor_list = []
        
        # Spawn vehicle
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = None
        while self.vehicle is None:
            try:
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            except:
                pass
        self.actor_list.append(self.vehicle)

        self.initial_location = self.vehicle.get_location()

        # Setup camera
        self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.sem_cam.set_attribute("fov", "90")
        
        camera_init_trans = carla.Transform(
            carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA_POS_X)
        )
        self.sensor = self.world.spawn_actor(self.sem_cam, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        # Initial control
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)

        # Apply random yaw
        angle_adj = random.randrange(-SPIN, SPIN, 1)
        trans = self.vehicle.get_transform()
        trans.rotation.yaw = trans.rotation.yaw + angle_adj
        self.vehicle.set_transform(trans)

        # Setup sensors
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lanesensor = self.world.spawn_actor(lanesensor, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.lanesensor)
        self.lanesensor.listen(lambda event: self.lane_data(event))

        # Wait for camera
        while self.front_camera is None:
            time.sleep(0.01)

        # Initialize episode variables
        self.episode_start = time.time()
        self.steering_lock = False
        self.steering_lock_start = None
        self.step_counter = 0
        
        # Initial control
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        
        # Process initial image
        self.image_for_CNN = self.apply_cnn(
            self.front_camera[self.height_from:, self.width_from:self.width_to]
        )
        
        return self.image_for_CNN

    def cleanup(self):
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        cv2.destroyAllWindows()

    def process_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3]
        self.front_camera = i

    def apply_cnn(self, im):
        img = np.float32(im)
        img = img / 255
        img = np.expand_dims(img, axis=0)
        cnn_applied = self.cnn_model([img, 0], training=False)
        cnn_applied = np.squeeze(cnn_applied)
        return cnn_applied

    def collision_data(self, event):
        self.collision_hist.append(event)

    def lane_data(self, event):
        self.lane_invade_hist.append(event)
