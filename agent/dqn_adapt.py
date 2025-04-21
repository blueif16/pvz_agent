import os
import random
import time
import socket
import json
import struct
import numpy as np
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import namedtuple, deque
import subprocess
import atexit
import signal
import sys
from datetime import datetime
from replayBuffer import ReplayBuffer
import gymnasium as gym

# Constants from config
N_LANES = 5
LANE_LENGTH = 9
MAX_FRAMES = 10000
HP_NORM = 1
SUN_NORM = 200

def sum_onehot(grid):
    """Convert grid to one-hot encoding sum"""
    return torch.cat([torch.sum(grid==(i+1), axis=-1).unsqueeze(-1) for i in range(4)], axis=-1)

class ZombieNet(nn.Module):
    """Extract features from zombie state."""
    def __init__(self, input_size, output_size=48):
        super(ZombieNet, self).__init__()
        
        # Increased output size for better feature representation
        self.output_size = output_size
        
        # First create a linear layer to handle flattened zombie grid
        self.lane_conv = nn.Sequential(
            nn.Linear(N_LANES * LANE_LENGTH, N_LANES * LANE_LENGTH),  # Input is always the flattened 5x9 grid
            nn.ReLU(),
            nn.LayerNorm(N_LANES * LANE_LENGTH)
        )
        
        # Add convolutional layers to detect spatial patterns
        self.conv_layers = nn.Sequential(
            nn.Conv1d(N_LANES, 16, kernel_size=3, padding=1),  # Detect local patterns within lanes
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, padding=1),       # Further pattern detection
            nn.ReLU(),
        )
        
        # Add attention mechanism to focus on important lanes
        self.lane_attention = nn.Sequential(
            nn.Linear(LANE_LENGTH, 1),
            nn.Softmax(dim=1)
        )
        
        # Feature combination layers
        self.fc_combine = nn.Sequential(
            nn.Linear(16 * LANE_LENGTH + N_LANES, 64),  # Combine conv features with lane attention
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Ensure x is flattened to the right shape
        batch_size = x.size(0)
        
        # Reshape input to grid for processing if needed
        if x.dim() > 2:  # If input is 3D (batch, height, width)
            x = x.view(batch_size, -1)  # Flatten to (batch, height*width)
        
        # Make sure x has the right size
        if x.size(1) != N_LANES * LANE_LENGTH:
            # If not the right size, reshape or pad as needed
            x = x.view(batch_size, N_LANES * LANE_LENGTH)
        
        # Extract grid features
        grid_features = self.lane_conv(x)
        grid_features = grid_features.view(batch_size, N_LANES, LANE_LENGTH)
        
        # Apply convolutions to detect patterns within and across lanes
        conv_features = self.conv_layers(grid_features)  # Shape: [batch, 16, LANE_LENGTH]
        
        # Calculate lane importance using attention
        lane_importance = self.lane_attention(grid_features)  # Shape: [batch, N_LANES, 1]
        
        # Flatten conv features for final processing
        flat_conv = conv_features.view(batch_size, -1)
        flat_attn = lane_importance.squeeze(-1)  # Lane importance scores
        
        # Combine features with attention information
        combined = torch.cat([flat_conv, flat_attn], dim=1)
        output = self.fc_combine(combined)
        
        return output

class HealthProcessor(nn.Module):
    """Network to process plant health data and handle zombie data when needed"""
    def __init__(self, use_zombienet=False, grid_size=None):
        super(HealthProcessor, self).__init__()
        self.use_zombienet = use_zombienet
        self._grid_size = grid_size
        self.game_info_size = 6
        
        if use_zombienet:
            self.zombienet_output_size = 48  # Enhanced feature extraction
            # The input size for ZombieNet is always the flattened zombie grid size (45)
            zombie_grid_size = N_LANES * LANE_LENGTH
            self.zombienet = ZombieNet(input_size=zombie_grid_size, output_size=self.zombienet_output_size)
            # Calculate input size: plant grid + plant health grid + processed zombie features + game info
            self.n_inputs = 2 * self._grid_size + self.zombienet_output_size + self.game_info_size
        else:
            # Simple health processor
            self.fc = nn.Linear(self._grid_size, 2)
            self.n_inputs = 2 * self._grid_size + self._grid_size + self.game_info_size
        
    def forward(self, plant_grid, health_grid, zombie_grid, game_info):
        # Ensure all inputs are flattened
        batch_size = plant_grid.size(0)
        
        # Flatten grids if they are not already flat
        if plant_grid.dim() > 2:
            plant_grid = plant_grid.reshape(batch_size, -1)
        if health_grid.dim() > 2:
            health_grid = health_grid.reshape(batch_size, -1)
        if zombie_grid.dim() > 2:
            zombie_grid = zombie_grid.reshape(batch_size, -1)
        
        # Process zombie grid with ZombieNet if enabled
        if self.use_zombienet:
            # Process zombie grid
            zombie_features = self.zombienet(zombie_grid)
            # Combine all features
            return torch.cat([plant_grid, health_grid, zombie_features, game_info], dim=1)
        else:
            # Process health grid with simple network
            processed_health = self.fc(health_grid)
            # Combine all features
            return torch.cat([plant_grid, processed_health, zombie_grid, game_info], dim=1)

class DQNet(nn.Module):
    """The Deep-Q Network with multiple improvements."""
    def __init__(self, state_shape, action_shape, device, dueling=True, use_zombienet=True, learning_rate=1e-4):
        super(DQNet, self).__init__()
        
        self.dueling = dueling
        self.use_zombienet = use_zombienet
        self.device = device
        self.action_shape = action_shape
        
        # Define grid size for input processing
        self._grid_size = N_LANES * LANE_LENGTH  # 5 * 9 = 45
        
        # Process grid with plant and health info
        self.health_processor = HealthProcessor(use_zombienet=use_zombienet, grid_size=self._grid_size)
        
        # Feature extraction layers
        self.feature_size = 256  # Increased from 128 for richer feature representation
        
        # Add a series of more powerful feature extraction layers
        self.features = nn.Sequential(
            nn.Linear(self.health_processor.n_inputs, 128),
            nn.ReLU(),
            nn.LayerNorm(128),  # Normalization for more stable learning
            nn.Linear(128, self.feature_size),
            nn.ReLU(),
            nn.LayerNorm(self.feature_size),
        )
        
        # Zombie state attention mechanism
        if self.use_zombienet:
            # Attention mechanism to focus on important aspects of the zombie state
            self.zombie_attention = nn.Sequential(
                nn.Linear(self.feature_size, N_LANES),
                nn.Softmax(dim=1)
            )
            
            # Context vector computation
            self.zombie_context = nn.Sequential(
                nn.Linear(self.feature_size + N_LANES, self.feature_size),
                nn.ReLU(),
                nn.LayerNorm(self.feature_size)
            )
        
        # Dueling networks if enabled
        if self.dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(self.feature_size, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(self.feature_size, 128),
                nn.ReLU(),
                nn.Linear(128, action_shape)
            )
        else:
            self.output = nn.Sequential(
                nn.Linear(self.feature_size, 128),
                nn.ReLU(),
                nn.Linear(128, action_shape)
            )
            
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, plant_grid, health_grid, zombie_grid, sun, zombie_count=None, game_info=None):
        # Ensure the batch dimension is preserved
        batch_size = plant_grid.size(0)
        
        # Combine sun and other game info if available
        if game_info is None:
            game_info = torch.zeros(batch_size, 6, device=self.device)
            game_info[:, 0] = sun
        
        # Process health and plant grids
        x = self.health_processor(plant_grid, health_grid, zombie_grid, game_info)
        
        # Extract features
        features = self.features(x)
        
        # Apply zombie attention if using ZombieNet
        if self.use_zombienet:
            # Calculate attention weights for different aspects of zombie state
            attention_weights = self.zombie_attention(features)
            
            # Create context vector using attention
            context = torch.cat([features, attention_weights], dim=1)
            features = self.zombie_context(context)
        
        # Apply dueling architecture if enabled
        if self.dueling:
            values = self.value_stream(features)
            advantages = self.advantage_stream(features)
            
            # Combine using dueling formula: Q = V + (A - mean(A))
            qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            qvals = self.output(features)
            
        return qvals

class QNetwork_DQN(nn.Module):
    """Q-Network for DQN implementation based on second file architecture and adapted for RewardCounter's state format"""
    def __init__(self, action_space_size=182, epsilon=0.05, learning_rate=1e-3, device='cpu', use_zombienet=True, use_gridnet=True):
        super(QNetwork_DQN, self).__init__()
        self.device = device

        # Based on RewardCounter.processGameState:
        # Format: [tensor (3x5x9 flattened), gameInfo (6 values)]
        # Layer 0: Plant types (5x9)
        # Layer 1: Plant health (5x9)
        # Layer 2: Zombie health (5x9)
        # Game info: sun score, sun count, 4 card cooldowns (6 values)

        self._grid_size = N_LANES * LANE_LENGTH  # 5 * 9 = 45
        self.action_space_size = action_space_size
        self.actions = np.arange(action_space_size)
        self.learning_rate = learning_rate
        
        # For proper layer extraction
        self.plant_layer_size = self._grid_size  # 45 cells for plant types
        self.health_layer_size = self._grid_size  # 45 cells for plant health
        self.zombie_layer_size = self._grid_size  # 45 cells for zombie health
        self.game_info_size = 6  # sun score, sun count, 4 card cooldowns
        
        # Calculate initial input size
        self.n_inputs = 3 * self._grid_size + self.game_info_size

        # Process zombie grid with ZombieNet - larger output size as it's more important
        self.use_zombienet = use_zombienet
        if use_zombienet:
            self.zombienet_output_size = 20  # Increased from 1 to 10 for better representation
            self.zombienet = ZombieNet(input_size=self.zombienet_output_size)
            # Replace zombie grid with zombienet output
            self.n_inputs = 2 * self._grid_size + N_LANES * self.zombienet_output_size + self.game_info_size

        # Process plant grid with GridNet
        self.use_gridnet = use_gridnet
        if use_gridnet:
            self.gridnet_output_size = 5  # Increased from 4 to 8 for better representation
            self.gridnet = nn.Linear(self._grid_size, self.gridnet_output_size)
            # Replace plant grid with gridnet output
            self.n_inputs = self.gridnet_output_size + self._grid_size + (N_LANES * self.zombienet_output_size if use_zombienet else self._grid_size) + self.game_info_size
        
        # Process plant health with simplified network
        self.use_health_processor = True
        if self.use_health_processor:
            self.health_output_size = 2  # Reduced as plant health is less important
            self.health_processor = HealthProcessor(use_zombienet=use_zombienet)
            # Update input size to use processed health
            self.n_inputs = (self.gridnet_output_size if use_gridnet else self._grid_size) + \
                            self.health_output_size + \
                            (N_LANES * self.zombienet_output_size if use_zombienet else self._grid_size) + \
                            self.game_info_size

        # Set up main network with wider layers for better representation
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(256, 192, bias=True),  # Increased from 128 to 192
            nn.LeakyReLU(),
            nn.Linear(192, self.action_space_size, bias=True))

        # Set to GPU if cuda is specified
        if self.device == 'cuda':
            self.network.cuda()
            
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                          lr=self.learning_rate)
        
    def decide_action(self, state, mask, epsilon):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() < epsilon:
            valid_actions = np.where(mask)[0]
            if len(valid_actions) == 0:
                return 0  # Default to ACTION_DO_NOTHING if no valid actions
            action = np.random.choice(valid_actions)
        else:
            action = self.get_greedy_action(state, mask)
        return action
    
    def get_greedy_action(self, state, mask):
        """Choose the greedy action respecting the action mask"""
        qvals = self.get_qvals(state)
        
        # Convert to numpy for masking if it's a tensor
        if isinstance(qvals, torch.Tensor):
            qvals_np = qvals.detach().cpu().numpy()
        else:
            qvals_np = qvals
            
        # Apply mask - set invalid actions to minimum value
        qvals_np[np.logical_not(mask)] = float('-inf')
        
        # Get best valid action
        if np.all(qvals_np == float('-inf')):
            return 0  # Default to ACTION_DO_NOTHING if all actions are invalid
        return np.argmax(qvals_np)

    def get_qvals(self, state):
        """
        Get Q values for the given state
        
        State format from RewardCounter.processGameState:
        - First 45 values: Plant types (5x9 grid)
        - Next 45 values: Plant health (5x9 grid)
        - Next 45 values: Zombie health (5x9 grid)
        - 6 game info values: sun score, sun count, 4 card cooldowns
        """
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state_t = torch.FloatTensor(state).to(device=self.device)
        else:
            state_t = state
            
        # Handle batch dimension
        batch_mode = len(state_t.shape) > 1
        if not batch_mode:
            state_t = state_t.unsqueeze(0)  # Add batch dimension
            
        # Extract layers from the state tensor
        plant_types = state_t[:, :self.plant_layer_size]  # Plant types
        plant_health = state_t[:, self.plant_layer_size:2*self.plant_layer_size]  # Plant health
        zombie_health = state_t[:, 2*self.plant_layer_size:3*self.plant_layer_size]  # Zombie health
        game_info = state_t[:, 3*self.plant_layer_size:]  # Game info (sun score, sun count, cooldowns)
        
        # Reshape zombie grid for zombienet (batch_size, n_lanes, lane_length)
        zombie_grid_reshaped = zombie_health.view(-1, N_LANES, LANE_LENGTH)
        
        # Process layers
        processed_layers = []
        
        # Process plant grid with gridnet if enabled
        if self.use_gridnet:
            processed_plant_grid = self.gridnet(plant_types)
            processed_layers.append(processed_plant_grid)
        else:
            processed_layers.append(plant_types)
            
        # Process plant health with health processor or use as is
        if self.use_health_processor:
            processed_plant_health = self.health_processor(plant_types, plant_health, zombie_health, game_info)
            processed_layers.append(processed_plant_health)
        else:
            processed_layers.append(plant_health)
        
        # Process zombie grid with zombienet if enabled
        if self.use_zombienet:
            # Process each lane
            zombie_outputs = []
            for lane in range(N_LANES):
                lane_data = zombie_grid_reshaped[:, lane, :]
                lane_output = self.zombienet(lane_data)
                zombie_outputs.append(lane_output)
            # Concatenate outputs from all lanes
            processed_zombie_grid = torch.cat(zombie_outputs, dim=1)
            processed_layers.append(processed_zombie_grid)
        else:
            processed_layers.append(zombie_health)
            
        # Add game info
        processed_layers.append(game_info)
        
        # Concatenate all processed layers
        combined_state = torch.cat(processed_layers, dim=1)
        
        # Get Q-values
        q_values = self.network(combined_state)
        
        # Remove batch dimension if input wasn't batched
        if not batch_mode:
            q_values = q_values.squeeze(0)
            
        return q_values

    def forward(self, tensor_input, game_info_input, action_mask=None):
        """
        Forward pass for the network - for compatibility with the original architecture
        
        Parameters:
        - tensor_input: 4D tensor of shape (batch_size, 3, 5, 9) containing:
                       - Channel 0: Plant types
                       - Channel 1: Plant health
                       - Channel 2: Zombie health
        - game_info_input: 2D tensor of shape (batch_size, 6) containing game info
        - action_mask: Optional mask for valid actions
        
        Returns:
        - Q-values for each action
        """
        batch_size = tensor_input.shape[0]
        
        # Reshape tensor input from (batch, 3, 5, 9) to match our expected format
        # Extract the three channels and flatten each into 45 values
        plant_types = tensor_input[:, 0, :, :].reshape(batch_size, -1)     # (batch, 45)
        plant_health = tensor_input[:, 1, :, :].reshape(batch_size, -1)    # (batch, 45)
        zombie_health = tensor_input[:, 2, :, :].reshape(batch_size, -1)   # (batch, 45)
        
        # Combine into state format: [plant_types, plant_health, zombie_health, game_info]
        state = torch.cat([plant_types, plant_health, zombie_health, game_info_input], dim=1)
        
        # Get Q-values
        q_values = self.get_qvals(state)
        
        # Apply action mask if provided
        if action_mask is not None:
            # Set Q-values of invalid actions to a very low value
            q_values = q_values + (action_mask - 1) * 1e9
            
        return q_values


class GameConnector:
    """Manages communication with the Java game connector."""
    def __init__(self, port=5555):
        self.port = port
        self.socket = None
        self.running = False
        self.connector_process = None
        
    def start(self):
        """Start the connector process and establish connection."""
        # Launch the connector process
        try:
            cmd = ["java", "-jar", "../PlantsVsZombies.jar", 
                   "--connector", 
                   f"--port={self.port}"]
            
            print(f"Launching connector on port {self.port}")
            self.connector_process = subprocess.Popen(cmd)
            
            # Give the connector time to start
            time.sleep(5)  # Increased from 2 to 5 seconds
            
            # Connect to the connector
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # Set a timeout for the connection attempt
            self.socket.settimeout(10)  # 10 second timeout
            
            # Try to connect multiple times
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    print(f"Connecting to connector on localhost:{self.port} (attempt {attempt+1}/{max_attempts})...")
                    self.socket.connect(('localhost', self.port))
                    self.running = True
                    print(f"Connected to connector on port {self.port}")
                    
                    # Reset timeout to default after successful connection
                    self.socket.settimeout(None)
                    
                    return True
                except socket.error as e:
                    if attempt < max_attempts - 1:
                        print(f"Connection attempt failed: {e}. Retrying in 2 seconds...")
                        time.sleep(2)
                    else:
                        raise e
        except Exception as e:
            print(f"Error starting connector: {e}")
            return False
    
    def create_envs(self, num_envs, max_steps=10000, training_mode=True):
        """Request the connector to create game environments."""
        try:
            # Create command message
            command = {
                'type': 'create_envs',
                'num_envs': num_envs,
                'max_steps': max_steps,
                'training_mode': training_mode
            }
            
            # Send command
            self.send_data(command)
            
            # Wait for response
            response = self.receive_data()
            
            if response and response.get('status') == 'success':
                print(f"Created {num_envs} game environments with max_steps={max_steps}, training_mode={training_mode}")
                return True
            else:
                print(f"Failed to create environments: {response}")
                return False
        except Exception as e:
            print(f"Error creating environments: {e}")
            return False
    
    def reset(self):
        """Reset all game environments."""
        try:
            # Create command message
            command = {
                'type': 'reset'
            }
            
            # Send command
            self.send_data(command)
            
            # Wait for response
            response = self.receive_data()
            
            if response and response.get('status') == 'success':
                states = response.get('states')
                action_masks = response.get('action_masks')
                
                # Convert to numpy arrays
                states_np = [np.array(state, dtype=np.float32) for state in states]
                action_masks_np = [np.array(mask, dtype=np.float32) for mask in action_masks]
                
                return states_np, action_masks_np
            else:
                print(f"Failed to reset environments: {response}")
                return None, None
        except Exception as e:
            print(f"Error resetting environments: {e}")
            return None, None
    
    def step(self, actions):
        """Execute steps in all game environments."""
        try:
            # Create command message
            command = {
                'type': 'step',
                'actions': actions
            }
            
            # Send command
            self.send_data(command)
            
            # Wait for response
            response = self.receive_data()
            
            if response and response.get('status') == 'success':
                # Extract data from response
                states = response.get('states')
                action_masks = response.get('action_masks')
                rewards = response.get('rewards')
                dones = response.get('dones')
                truncateds = response.get('truncateds')
                final_observations = response.get('final_observations')
                
                # Convert to numpy arrays
                states_np = [np.array(state, dtype=np.float32) if state is not None else None for state in states]
                action_masks_np = [np.array(mask, dtype=np.float32) if mask is not None else None for mask in action_masks]
                rewards_np = np.array(rewards, dtype=np.float32)
                dones_np = np.array(dones, dtype=bool)
                truncateds_np = np.array(truncateds, dtype=bool)
                final_obs_np = [np.array(obs, dtype=np.float32) if obs is not None else None for obs in final_observations]
                
                return states_np, action_masks_np, rewards_np, dones_np, truncateds_np, final_obs_np
            else:
                print(f"Failed to step environments: {response}")
                return None, None, None, None, None, None
        except Exception as e:
            print(f"Error stepping environments: {e}")
            return None, None, None, None, None, None
    
    def send_data(self, data):
        """Send data to the connector, properly handling NumPy data types."""
        try:
            # Convert numpy arrays and scalars to Python native types
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32, np.float16)):
                    return float(obj)
                elif isinstance(obj, (np.bool_)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list) or isinstance(obj, tuple):
                    return [convert_numpy(i) for i in obj]
                else:
                    return obj
            
            data = convert_numpy(data)
            
            # Convert to JSON
            json_data = json.dumps(data).encode('utf-8')
            
            # Send the data length first (as 4 bytes)
            data_length = len(json_data)
            self.socket.sendall(struct.pack('!I', data_length))
            
            # Then send the actual data
            self.socket.sendall(json_data)
            return True
        except Exception as e:
            print(f"Error sending data: {e}")
            self.running = False
            return False
    
    def receive_data(self):
        """Receive data from the connector."""
        try:
            # First receive the data length (4 bytes)
            length_bytes = self.socket.recv(4)
            if not length_bytes:
                return None
            
            data_length = struct.unpack('!I', length_bytes)[0]
            
            # Then receive the actual data
            data_bytes = b''
            bytes_received = 0
            
            while bytes_received < data_length:
                chunk = self.socket.recv(min(4096, data_length - bytes_received))
                if not chunk:
                    return None
                data_bytes += chunk
                bytes_received += len(chunk)
            
            # Parse JSON data
            json_data = data_bytes.decode('utf-8')
            data = json.loads(json_data)
            return data
        except Exception as e:
            print(f"Error receiving data: {e}")
            self.running = False
            return None
    
    def stop(self):
        """Stop the connector."""
        try:
            if self.running:
                # Send stop command
                command = {
                    'type': 'close'
                }
                self.send_data(command)
            
            if self.socket:
                self.socket.close()
                self.socket = None
            
            if self.connector_process:
                self.connector_process.terminate()
                try:
                    self.connector_process.wait(timeout=5)
                except:
                    self.connector_process.kill()
                self.connector_process = None
            
            self.running = False
            print(f"Connector stopped")
            return True
        except Exception as e:
            print(f"Error stopping connector: {e}")
            return False

class Threshold:
    """Manages epsilon value decay for exploration"""
    def __init__(self, seq_length=100000, start_epsilon=1.0, end_epsilon=0.05, interpolation="exponential"):
        self.seq_length = seq_length
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.interpolation = interpolation
        
    def epsilon(self, step):
        """Calculate epsilon value for the given step"""
        if self.interpolation == "exponential":
            # Exponential decay
            decay_rate = -np.log(self.end_epsilon / self.start_epsilon) / self.seq_length
            return self.start_epsilon * np.exp(-decay_rate * step)
        elif self.interpolation == "linear":
            # Linear decay
            return max(self.end_epsilon, self.start_epsilon - step * (self.start_epsilon - self.end_epsilon) / self.seq_length)
        elif self.interpolation == "sinusoidal":
            # Sinusoidal decay
            return max(self.end_epsilon, 0.5 * (self.start_epsilon - self.end_epsilon) * 
                      (1 + np.cos(step * np.pi / self.seq_length)) + self.end_epsilon)
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation}")

class experienceReplayBuffer_DQN:
    """Experience replay buffer for DQN"""
    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer', 
            field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size,
                                   replace=False)
        # Use asterisk operator to unpack deque 
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(
            self.Buffer(state, action, reward, done, next_state))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in

class PvZDQNAgent:
    """DQN agent for Plants vs Zombies using the new architecture."""
    def __init__(self, num_envs=1, port=5555, fast_training=True):
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize connector
        self.connector = GameConnector(port=port)
        
        # Training parameters
        self.num_envs = num_envs
        self.fast_training = fast_training
        self.action_space_size = 182  # 4*5*9 + 2 (do nothing + collect sun)
        self.buffer_size = 50000      # Increased from 10000 to 50000 for more stable learning
        self.batch_size = 256         # Increased from 128 to 256 for better gradient estimation
        self.gamma = 0.995            # Increased from 0.99 to 0.995 for longer-term rewards
        self.learning_rate = 1e-4     # Decreased from 2.5e-4 to 1e-4 for more stable learning
        
        self.epsilon_start = 1.0
        self.epsilon_final = 0.05
        self.epsilon_decay = 200000   # Increased from 100000 to 200000 for more exploration
        self.target_update_freq = 1000 # Decreased from 2000 to 1000 for more frequent target updates
        self.global_step = 0
        self.log_freq = 100          # Frequency for logging metrics

        # Training parameters
        self.learning_starts = 5000   # Increased from 4000 to 5000 to collect more initial experiences
        self.train_frequency = 4      # Decreased from 500 to 4 for more frequent updates (every 4 steps)
        self.should_save_model = True
        self.save_freq = 10000        # Increased from 5000 to 10000 to save less frequently

        self.step_interval = 0.00001

        # Create threshold for epsilon decay
        self.threshold = Threshold(
            seq_length=self.epsilon_decay,
            start_epsilon=self.epsilon_start,
            end_epsilon=self.epsilon_final,
            interpolation="exponential"
        )
        self.epsilon = self.epsilon_start
        
        # Initialize tracking variables
        self.runs_cumulative_reward = []
        self.recent_losses = []       # Track recent losses for logging
        self.episode_rewards = [[0] for _ in range(self.num_envs)]
        self.episode_lengths = [[0] for _ in range(self.num_envs)]  # Track steps per episode
        self.completed_episodes = 0   # Track completed episodes
        self.writer = SummaryWriter(f"runs/pvz_dqn_{int(time.time())}")
        self.total_steps = 0
        
        # Episode length tracking
        self.all_episode_lengths = []  # Store lengths of all completed episodes
        self.moving_avg_length = deque(maxlen=100)  # Moving average of last 100 episode lengths
        
        # Zombie proximity penalty tracking
        self.zombie_penalties = []  # Store all zombie penalties for analysis
        self.episode_penalties = [[0] for _ in range(self.num_envs)]  # Track penalties per episode
        self.recent_penalties = deque(maxlen=100)  # Track recent penalties for monitoring
        
        # Initialize replay buffer
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3*5*9 + 6,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.action_space_size)
        
        # Initialize buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            n_envs=self.num_envs,
            action_space=self.action_space,
            device=self.device,
            handle_timeout_termination=False,
        )
        
        # Initialize networks
        self.q_network = None
        self.target_network = None

    def _convert_state_batch_for_network(self, states):
        """
        Convert batch of states from replay buffer to the format expected by the network.
        
        Args:
            states: Batch of states from the replay buffer (batch_size, state_dim)
                where state_dim = 3*grid_size + game_info_size
        
        Returns:
            tensor_states: Tensor input of shape (batch_size, 3, 5, 9)
            game_info_states: Game info of shape (batch_size, 6)
        """
        batch_size = states.shape[0]
        grid_size = N_LANES * LANE_LENGTH  # 5 * 9 = 45
        
        # Extract components from each state
        plant_types = states[:, :grid_size]  # First 45 values: Plant types
        plant_health = states[:, grid_size:2*grid_size]  # Next 45 values: Plant health
        zombie_health = states[:, 2*grid_size:3*grid_size]  # Next 45 values: Zombie health
        game_info = states[:, 3*grid_size:]  # Last 6 values: Game info
        
        # Normalize if needed
        plant_health = plant_health / HP_NORM if HP_NORM > 1 else plant_health
        zombie_health = zombie_health / HP_NORM if HP_NORM > 1 else zombie_health
        
        # Create a copy of game_info for normalization to avoid modifying the original
        game_info_normalized = game_info.clone()
        
        # Normalize sun score (first element of game_info)
        if SUN_NORM > 1:
            game_info_normalized[:, 0] = game_info[:, 0] / SUN_NORM
        
        # Reshape each component into grids
        plant_types_grid = plant_types.reshape(batch_size, 5, 9)
        plant_health_grid = plant_health.reshape(batch_size, 5, 9)
        zombie_health_grid = zombie_health.reshape(batch_size, 5, 9)
        
        # Stack the grids to form a batch of 3x5x9 tensors
        tensor_states = torch.stack([plant_types_grid, plant_health_grid, zombie_health_grid], dim=1)
        
        return tensor_states, game_info_normalized
    
    def initialize_networks(self):
        """
        Initialize the Q-network and target network using architecture adapted for
        the Plants vs Zombies game state format from RewardCounter.
        """
        # Action space size = 4 plant types * 5 lanes * 9 columns + 2 special actions
        # As defined in RewardCounter.ACTION_SPACE_SIZE
        action_space_size = 4 * 5 * 9 + 2
        
        # Ensure it matches our cached value
        if self.action_space_size != action_space_size:
            print(f"Warning: Adjusting action space size from {self.action_space_size} to {action_space_size}")
            self.action_space_size = action_space_size
        
        # Create the Q-network with the adapted architecture
        self.q_network = DQNet(
            state_shape=(3, 5, 9),
            action_shape=self.action_space_size,
            device=self.device,
            dueling=True,
            use_zombienet=True,
            learning_rate=self.learning_rate
        ).to(self.device)
        
        # Create the target network with the same architecture
        self.target_network = DQNet(
            state_shape=(3, 5, 9),
            action_shape=self.action_space_size,
            device=self.device,
            dueling=True,
            use_zombienet=True,
            learning_rate=self.learning_rate
        ).to(self.device)
        
        # Initialize target network with the same weights as Q-network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        print(f"Networks initialized with action space size: {self.action_space_size}")
    
    def preprocess_state(self, state):
        """
        Unpack the flattened state from Java into tensor and game info components
        based on RewardCounter.processGameState format.
        
        Args:
            state: 1D array containing:
                - Plant types (5x9 = 45 values)
                - Plant health (5x9 = 45 values)
                - Zombie health (5x9 = 45 values)
                - Game info (6 values: sun score, sun count, 4 card cooldowns)
            
        Returns:
            tensor: 3D tensor of shape (3, 5, 9)
            game_info: 1D array of length 6
        """
        if state is None:
            # Return zeros if state is None (error case)
            return np.zeros((3, 5, 9)), np.zeros(6)
        
        # Convert to numpy array if it's not already
        state_array = np.array(state, dtype=np.float32)
        
        grid_size = N_LANES * LANE_LENGTH  # 5 * 9 = 45
        
        # Extract components based on RewardCounter.processGameState format
        plant_types = state_array[:grid_size]  # First 45 values: Plant types
        plant_health = state_array[grid_size:2*grid_size]  # Next 45 values: Plant health
        zombie_health = state_array[2*grid_size:3*grid_size]  # Next 45 values: Zombie health
        game_info = state_array[3*grid_size:]  # Last 6 values: Game info
        
        # Apply normalization
        # We keep plant types as is (they're already categorical)
        # Normalize health values if they can exceed 1.0
        plant_health = plant_health / HP_NORM if HP_NORM > 1 else plant_health
        zombie_health = zombie_health / HP_NORM if HP_NORM > 1 else zombie_health
        
        # Normalize sun score (first element of game_info)
        if SUN_NORM > 1 and len(game_info) > 0:
            game_info[0] = game_info[0] / SUN_NORM
        
        # Reshape each component into a 5x9 grid
        plant_types_grid = plant_types.reshape(5, 9)
        plant_health_grid = plant_health.reshape(5, 9)
        zombie_health_grid = zombie_health.reshape(5, 9)
        
        # Stack the grids to form a 3x5x9 tensor
        tensor = np.stack([plant_types_grid, plant_health_grid, zombie_health_grid], axis=0)
        
        return tensor, game_info
    
    def select_action(self, state, action_mask):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: State from RewardCounter.processGameState
            action_mask: Binary mask of valid actions (1 = valid, 0 = invalid)
        
        Returns:
            action_index: Integer representing the selected action
        """
        # Convert state data to tensors
        tensor, game_info = self.preprocess_state(state)
        
        # Convert action_mask to boolean array
        action_mask_bool = np.array(action_mask, dtype=bool)
        
        # Get valid actions
        valid_actions = np.where(action_mask_bool)[0]
        
        if len(valid_actions) == 0:
            return 0  # Return ACTION_DO_NOTHING if no valid actions
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action from valid actions
            return np.random.choice(valid_actions)
        else:
            # Get Q-values
            with torch.no_grad():
                # Convert inputs to tensors
                tensor_tensor = torch.FloatTensor(tensor).unsqueeze(0).to(self.device)
                game_info_tensor = torch.FloatTensor(game_info).unsqueeze(0).to(self.device)
                
                # Extract the tensor components needed for the network
                plant_types = tensor_tensor[:, 0]  # Plant types grid
                plant_health = tensor_tensor[:, 1]  # Plant health grid
                zombie_health = tensor_tensor[:, 2]  # Zombie health grid
                sun_score = game_info_tensor[:, 0]  # Sun score is first element in game info
                
                # Flatten the grid tensors to match the network's expected input
                plant_types_flat = plant_types.reshape(plant_types.size(0), -1)
                plant_health_flat = plant_health.reshape(plant_health.size(0), -1)
                zombie_health_flat = zombie_health.reshape(zombie_health.size(0), -1)
                
                # Forward pass through the network with the required arguments
                q_values = self.q_network(plant_types_flat, plant_health_flat, zombie_health_flat, sun_score, game_info=game_info_tensor)
                
                # Mask invalid actions
                q_values = q_values.cpu().numpy()
                q_values[0][~action_mask_bool] = float('-inf')
                
                # Select best valid action
                return np.argmax(q_values[0])
    
    def select_action_render(self, state, action_mask):
        """
        Select the greedy action (for rendering/evaluation).
        
        Args:
            state: State from RewardCounter.processGameState
            action_mask: Binary mask of valid actions (1 = valid, 0 = invalid)
            
        Returns:
            action_index: Integer representing the selected action
        """
        # Convert state data to tensors
        tensor, game_info = self.preprocess_state(state)
        
        # Convert action_mask to boolean array
        action_mask_bool = np.array(action_mask, dtype=bool)
        
        # Get valid actions
        valid_actions = np.where(action_mask_bool)[0]
        
        if len(valid_actions) == 0:
            return 0  # Return ACTION_DO_NOTHING if no valid actions
        
        # Get Q-values
        with torch.no_grad():
            # Convert inputs to tensors
            tensor_tensor = torch.FloatTensor(tensor).unsqueeze(0).to(self.device)
            game_info_tensor = torch.FloatTensor(game_info).unsqueeze(0).to(self.device)
            
            # Extract the tensor components needed for the network
            plant_types = tensor_tensor[:, 0]  # Plant types grid
            plant_health = tensor_tensor[:, 1]  # Plant health grid
            zombie_health = tensor_tensor[:, 2]  # Zombie health grid
            sun_score = game_info_tensor[:, 0]  # Sun score is first element in game info
            
            # Flatten the grid tensors to match the network's expected input
            plant_types_flat = plant_types.reshape(plant_types.size(0), -1)
            plant_health_flat = plant_health.reshape(plant_health.size(0), -1)
            zombie_health_flat = zombie_health.reshape(zombie_health.size(0), -1)
            
            # Forward pass through the network with the required arguments
            q_values = self.q_network(plant_types_flat, plant_health_flat, zombie_health_flat, sun_score, game_info=game_info_tensor)
            
            # Mask invalid actions
            q_values = q_values.cpu().numpy()
            q_values[0][~action_mask_bool] = float('-inf')
            
            # Select best valid action
            return np.argmax(q_values[0])
    
    def _update_network(self):
        """Update the Q-network using experiences from the replay buffer."""
        # Sample from replay buffer
        data = self.replay_buffer.sample(self.batch_size)
        
        # Get batch data
        states = data.observations
        actions = data.actions
        rewards = data.rewards.flatten()
        next_states = data.next_observations
        dones = data.dones.flatten()
        
        # Get next action masks - directly from the sample data
        # This handles the difference between our custom and the built-in ReplayBuffer
        if hasattr(data, 'next_action_masks'):
            # If using the custom ReplayBuffer with built-in mask support
            next_action_masks = data.next_action_masks
        elif hasattr(data, 'infos'):
            # If using the original ReplayBuffer with infos
            next_action_masks = torch.stack([info.get("next_action_masks", torch.ones(self.action_space_size, dtype=torch.float32).to(self.device)) 
                                            for info in data.infos]).to(self.device)
        else:
            # Fallback: assume all actions are valid (though this shouldn't typically happen)
            print("Warning: Could not find action masks in replay buffer sample")
            next_action_masks = torch.ones((states.shape[0], self.action_space_size), dtype=torch.float32).to(self.device)

        # Process states and next_states
        batch_size = states.shape[0]
        
        # Convert inputs to the format expected by network forward method
        tensor_states, game_info_states = self._convert_state_batch_for_network(states)
        tensor_next_states, game_info_next_states = self._convert_state_batch_for_network(next_states)
        
        # Extract components for network
        plant_types = tensor_states[:, 0]  # Plant types grid
        plant_health = tensor_states[:, 1]  # Plant health grid
        zombie_health = tensor_states[:, 2]  # Zombie health grid
        sun_score = game_info_states[:, 0]  # Sun score is first element in game info
        
        # Next state components
        next_plant_types = tensor_next_states[:, 0]
        next_plant_health = tensor_next_states[:, 1]
        next_zombie_health = tensor_next_states[:, 2]
        next_sun_score = game_info_next_states[:, 0]
        
        # Flatten grid tensors to match network's expected input
        plant_types_flat = plant_types.reshape(batch_size, -1)
        plant_health_flat = plant_health.reshape(batch_size, -1)
        zombie_health_flat = zombie_health.reshape(batch_size, -1)
        
        # Similarly for next state
        next_plant_types_flat = next_plant_types.reshape(batch_size, -1)
        next_plant_health_flat = next_plant_health.reshape(batch_size, -1)
        next_zombie_health_flat = next_zombie_health.reshape(batch_size, -1)
        
        # Compute current Q-values
        current_q_values = self.q_network(plant_types_flat, plant_health_flat, zombie_health_flat, sun_score, game_info=game_info_states)
        state_action_values = current_q_values.gather(1, actions).squeeze()
        
        # Compute next state Q-values using the target network
        with torch.no_grad():
            # Get Q-values for next states
            next_q_values = self.target_network(next_plant_types_flat, next_plant_health_flat, next_zombie_health_flat, 
                                               next_sun_score, game_info=game_info_next_states)
            
            # Apply action masks to make invalid actions have very low Q-values
            next_q_values = next_q_values + (next_action_masks - 1) * 1e9
            
            # Get maximum Q-value for each next state
            max_next_q_values = next_q_values.max(1)[0]
        
        # Normalize rewards to mean 0, std 1, and optionally rescale
        if len(rewards) > 1:  # Only normalize if we have more than one sample
            reward_mean = rewards.mean()
            reward_std = rewards.std()
            # Add a small constant to avoid division by zero
            reward_std = torch.max(reward_std, torch.tensor(1e-6, device=self.device))
            
            # Normalize to mean 0, std 1
            normalized_rewards = (rewards - reward_mean) / reward_std
            
            # Optionally, rescale to a smaller range to prevent extreme values
            scale_factor = 1.0  # Adjust this if needed
            normalized_rewards = normalized_rewards * scale_factor
            
            # Log reward statistics periodically
            if self.global_step % 1000 == 0:
                raw_min = rewards.min().item()
                raw_max = rewards.max().item()
                raw_mean = reward_mean.item()
                raw_std = reward_std.item()
                norm_min = normalized_rewards.min().item()
                norm_max = normalized_rewards.max().item()
                
                print(f"Reward stats - Raw: min={raw_min:.2f}, max={raw_max:.2f}, mean={raw_mean:.2f}, std={raw_std:.2f}")
                print(f"Normalized: min={norm_min:.2f}, max={norm_max:.2f}")
                
                # Log to tensorboard
                self.writer.add_scalar("rewards/raw_min", raw_min, self.global_step)
                self.writer.add_scalar("rewards/raw_max", raw_max, self.global_step)
                self.writer.add_scalar("rewards/raw_mean", raw_mean, self.global_step)
                self.writer.add_scalar("rewards/raw_std", raw_std, self.global_step)
                self.writer.add_scalar("rewards/norm_min", norm_min, self.global_step)
                self.writer.add_scalar("rewards/norm_max", norm_max, self.global_step)
        else:
            normalized_rewards = rewards
        
        # Calculate target Q-values using normalized rewards
        target_q_values = normalized_rewards + (1 - dones) * self.gamma * max_next_q_values
    
        # Compute loss (Huber loss for stability)
        loss = F.smooth_l1_loss(state_action_values, target_q_values)
        loss_value = loss.item()
        
        # Store for logging
        self.recent_losses.append(loss_value)
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)
        
        # Log loss periodically
        if self.global_step % 10000 == 0:
            print(f"Loss: {loss_value:.6f}")
            
            # Log loss to file
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            loss_log_file = os.path.join(log_dir, "loss_log.txt")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(loss_log_file, "a") as f:
                f.write(f"{timestamp} | Step: {self.global_step} | Loss: {loss_value:.6f}")
                if len(self.recent_losses) >= 10:
                    avg_loss = np.mean(self.recent_losses[-10:])
                    f.write(f" | Avg Loss (10): {avg_loss:.6f}\n")
                else:
                    f.write("\n")
        
        # Optimize
        self.q_network.optimizer.zero_grad()
        loss.backward()
        # Clip gradients 
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.q_network.optimizer.step()
        
        # Log loss to tensorboard periodically
        if self.global_step % 100 == 0:
            self.writer.add_scalar("losses/q_loss", loss_value, self.global_step)
            self.writer.add_scalar("training/epsilon", self.epsilon, self.global_step)
    
    def calculate_zombie_proximity_penalty(self, state):
        """
        Calculate a penalty based on how close zombies are to the house.
        Closer zombies result in higher penalties.
        
        Args:
            state: State from RewardCounter.processGameState
            
        Returns:
            penalty: Negative value representing the penalty
        """
        if state is None:
            return 0.0
            
        # Extract zombie health grid from state
        # State format: [plant_types (45), plant_health (45), zombie_health (45), game_info (6)]
        grid_size = N_LANES * LANE_LENGTH  # 5 * 9 = 45
        zombie_health = state[2*grid_size:3*grid_size]
        plant_types = state[:grid_size]  # Plant types grid
        sun_score = state[3*grid_size]  # Sun score is first element of game info
        
        # Reshape to 5x9 grid (5 lanes, 9 columns)
        zombie_grid = np.array(zombie_health).reshape(N_LANES, LANE_LENGTH)
        plant_grid = np.array(plant_types).reshape(N_LANES, LANE_LENGTH)
        
        # Calculate penalty based on zombie positions
        # Column 0 is closest to house, column 8 is furthest
        total_penalty = 0.0
        urgency_bonus = 0.0
        
        # Define penalties for different distances - exponential scaling for more urgency
        # Higher penalties for zombies closer to the house
        proximity_weights = [5.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
        
        # Identify lanes with zombies and their defenses
        lanes_with_zombies = set()
        lanes_with_defenses = set()
        
        # Count zombies in each column for urgency assessment
        zombies_by_column = [0] * LANE_LENGTH
        
        for lane in range(N_LANES):
            has_defense = False
            
            # Check if lane has defensive plants
            for col in range(LANE_LENGTH):
                if plant_grid[lane, col] in [2, 3, 4]:  # Peashooter, FreezePeashooter, Wallnut
                    has_defense = True
                    lanes_with_defenses.add(lane)
                    break
            
            for col in range(LANE_LENGTH):
                if zombie_grid[lane, col] > 0:  # If there's a zombie at this position
                    lanes_with_zombies.add(lane)
                    zombies_by_column[col] += 1
                    
                    # Apply penalty based on proximity to house
                    # Column 0 (closest to house) gets highest penalty
                    penalty_factor = proximity_weights[col]
                    
                    # Scale penalty by zombie health (stronger zombies = higher penalty)
                    zombie_strength = min(zombie_grid[lane, col] / HP_NORM, 1.0) 
                    
                    # Add to total penalty
                    zombie_penalty = penalty_factor * zombie_strength
                    
                    # Extra penalty for undefended lanes
                    if not has_defense:
                        zombie_penalty *= 2.0
                        
                    total_penalty += zombie_penalty
                    
                    # Urgent columns (closest to house) need immediate attention
                    if col < 3 and not has_defense:
                        urgency_bonus += 1.0
        
        # Calculate the base penalty
        base_penalty = -0.2 * total_penalty
        
        # Additional strategic considerations
        strategic_signal = 0.0
        
        # 1. Urgent need to defend lanes with zombies but no defenses
        undefended_lanes = lanes_with_zombies - lanes_with_defenses
        if undefended_lanes and sun_score >= 100:  # Have enough sun for at least a peashooter
            strategic_signal -= len(undefended_lanes) * 0.5  # Penalty for not defending
        
        # 2. Front-loaded zombie columns need immediate attention
        if zombies_by_column[0] + zombies_by_column[1] > 0 and sun_score >= 50:
            strategic_signal -= urgency_bonus  # Urgent defense needed
            
        # 3. If we have many zombies incoming but little sun, collecting sun becomes more important
        if sum(zombies_by_column) > 3 and sun_score < 100:
            strategic_signal -= 0.3  # Encourage sun collection
        
        return base_penalty + strategic_signal
    
    def run(self, max_steps=100000, training_mode=True):
        """Start training with multiple environments."""
        # Start the connector
        if not self.connector.start():
            print("Failed to start connector")
            return
        
        # Create environments with higher max_steps to allow for longer episodes
        if not self.connector.create_envs(self.num_envs, max_steps=50000, training_mode=training_mode):
            print("Failed to create environments")
            return
        
        # Initialize networks if not done yet
        if self.q_network is None:
            self.initialize_networks()
        
        # Reset all environments to get initial states
        states, action_masks = self.connector.reset()
        if states is None:
            print("Failed to reset environments")
            return

        # For tracking critical states for learning
        recent_zombie_states = deque(maxlen=1000)  # Store states with zombies for prioritized replay
        state_priorities = {}  # Track state importance for learning
        
        # Curriculum phases
        curriculum_phase = 0  # 0: Basic planting, 1: Zombie defense, 2: Advanced strategy
        curriculum_thresholds = [200, 1000, 2000]  # Step thresholds for curriculum phases
        
        try:
            # Training loop
            while self.total_steps < max_steps:
                # Select actions for all environments
                actions = [self.select_action(state, mask) 
                        for state, mask in zip(states, action_masks)]
                
                # Execute actions in all environments
                try:
                    next_states, next_action_masks, rewards, dones, truncateds, final_observations = self.connector.step(actions)
                    
                    if next_states is None:
                        print("Failed to step environments, resetting...")
                        states, action_masks = self.connector.reset()
                        if states is None:
                            print("Failed to reset environments")
                            break
                        continue
                    
                    # Apply zombie proximity penalties to rewards
                    critical_state_indices = []  # Track which states are critical for learning
                    for i in range(self.num_envs):
                        if next_states[i] is not None:
                            # Calculate and apply penalty
                            zombie_penalty = self.calculate_zombie_proximity_penalty(next_states[i])
                            
                            # Identify critical states with zombies for prioritized replay
                            grid_size = N_LANES * LANE_LENGTH
                            zombie_health = next_states[i][2*grid_size:3*grid_size]
                            zombie_grid = np.array(zombie_health).reshape(N_LANES, LANE_LENGTH)
                            
                            # If state has zombies, prioritize it for replay
                            if np.sum(zombie_grid) > 0:
                                zombie_count = np.count_nonzero(zombie_grid)
                                front_zombies = np.sum(zombie_grid[:, :3])  # Zombies in first 3 columns
                                
                                # Store state hash and importance
                                state_hash = hash(str(next_states[i].round(2)))
                                
                                # Calculate priority based on zombie presence and proximity
                                priority = 1.0
                                
                                # Higher priority for states with zombies near the house
                                if front_zombies > 0:
                                    priority += 2.0
                                    critical_state_indices.append(i)
                                
                                # Higher priority for states with many zombies
                                if zombie_count > 3:
                                    priority += 1.0
                                    critical_state_indices.append(i)
                                
                                # Store state and priority for potential replay
                                state_priorities[state_hash] = priority
                                recent_zombie_states.append((next_states[i], next_action_masks[i], actions[i], rewards[i], dones[i]))
                            
                            rewards[i] += zombie_penalty
                            
                            # Track penalties
                            self.recent_penalties.append(zombie_penalty)
                            self.episode_penalties[i].append(zombie_penalty)
                            
                            # Log significant penalties (for debugging/monitoring)
                            if zombie_penalty < -0.3 and self.global_step % 100 == 0:
                                print(f"Applied zombie proximity penalty: {zombie_penalty:.4f}")
                
                    # Log average penalty every 100 steps
                    if self.global_step % 100 == 0 and len(self.recent_penalties) > 0:
                        avg_penalty = np.mean(self.recent_penalties)
                        self.writer.add_scalar("rewards/zombie_proximity_penalty", avg_penalty, self.global_step)
                    
                    # Process each environment's results
                    real_next_states = next_states.copy()
                    for i in range(self.num_envs):
                        if truncateds[i]:
                            real_next_states[i] = final_observations[i]
                    
                    # Add to replay buffer
                    try:
                        actions_array = np.array(actions).reshape(-1, 1)  # Reshape actions to work with multiple envs
                        self.replay_buffer.add(
                            np.array(states),
                            np.array(real_next_states),
                            actions_array,
                            np.array(rewards),
                            np.array(dones),
                            [{"next_action_masks": mask} for mask in next_action_masks]
                        )
                    except Exception as e:
                        print(f"Error adding to replay buffer: {e}")
                        print(f"States shape: {np.array(states).shape}")
                        print(f"Next states shape: {np.array(real_next_states).shape}")
                        print(f"Actions shape: {actions_array.shape}")
                        print(f"Rewards shape: {np.array(rewards).shape}")
                        print(f"Dones shape: {np.array(dones).shape}")
                        print(f"Action masks: {len(next_action_masks)}")
                        # Continue despite error
                        continue
                    
                    # Update curriculum phase based on steps
                    for threshold_idx, threshold in enumerate(curriculum_thresholds):
                        if self.global_step >= threshold and curriculum_phase == threshold_idx:
                            curriculum_phase += 1
                            print(f"Advancing to curriculum phase {curriculum_phase}")
                            
                            # Adjust learning parameters for each phase
                            if curriculum_phase == 1:  # Zombie defense phase
                                # More frequent updates to learn faster
                                self.train_frequency = 2
                            elif curriculum_phase == 2:  # Advanced strategy phase
                                # Lower epsilon to exploit more
                                self.epsilon = max(0.1, self.epsilon * 0.8)
                                
                    states = next_states
                    action_masks = next_action_masks

                    # Update global step count
                    self.global_step += self.num_envs
                    self.total_steps += self.num_envs
                    
                    # Update epsilon 
                    self.epsilon = self.threshold.epsilon(self.global_step)
                    
                    # Train the network if we have enough samples
                    if self.global_step > self.learning_starts and self.global_step % self.train_frequency == 0:
                        try:
                            self._update_network()
                        except Exception as e:
                            print(f"Error in _update_network: {e}")
                        
                        # Add extra training on critical zombie states if we're in curriculum phase 1+
                        if curriculum_phase >= 1 and recent_zombie_states and self.global_step % 20 == 0:
                            try:
                                # Sample a few high-priority zombie states for extra training
                                num_extra_samples = min(5, len(recent_zombie_states))
                                for _ in range(num_extra_samples):
                                    state, mask, action, reward, done = random.choice(recent_zombie_states)
                                    
                                    # Create arrays with correct environment dimensions
                                    state_array = np.array([state for _ in range(self.num_envs)])
                                    next_state_array = np.array([state for _ in range(self.num_envs)])
                                    action_array = np.array([action for _ in range(self.num_envs)]).reshape(-1, 1)
                                    reward_array = np.array([reward * 1.2 for _ in range(self.num_envs)])
                                    done_array = np.array([done for _ in range(self.num_envs)])
                                    mask_array = [{"next_action_masks": mask} for _ in range(self.num_envs)]
                                    
                                    self.replay_buffer.add(
                                        state_array,
                                        next_state_array,
                                        action_array,
                                        reward_array,
                                        done_array,
                                        mask_array
                                    )
                            except Exception as e:
                                print(f"Error in critical state training: {e}")
                                # Continue despite error
                                pass
                except Exception as e:
                    print(f"Error during environment step: {e}")
                    # Try to reset the environments to recover
                    try:
                        states, action_masks = self.connector.reset()
                        if states is None:
                            print("Failed to reset environments after error")
                            break
                        continue
                    except Exception as reset_error:
                        print(f"Error resetting environments: {reset_error}")
                        break
                
                # Update target network
                if self.global_step % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                    print(f"Target network updated at step {self.global_step} (curriculum phase: {curriculum_phase})")
                    
                    # Log average episode length if we have data
                    if len(self.moving_avg_length) > 0:
                        avg_length = np.mean(self.moving_avg_length)
                        print(f"Average episode length (last 100): {avg_length:.2f} steps")
                        
                    if self.should_save_model and self.global_step % self.save_freq == 0:
                        # Save new model
                        self.save_model(f"models/pvz_dqn_step{self.global_step}.pt")
                        
                        # Log training metrics
                        self._log_training_metrics()
                
                # Track episode length and detect episode completion
                for env_idx in range(self.num_envs):
                    # Increment episode length
                    self.episode_lengths[env_idx][-1] += 1
                    
                    # Add reward
                    self.episode_rewards[env_idx].append(rewards[env_idx])
                    
                    if dones[env_idx] or truncateds[env_idx]:
                        # Calculate total episode metrics
                        total_ep_reward = np.sum(self.episode_rewards[env_idx])
                        ep_length = self.episode_lengths[env_idx][-1]
                        total_ep_penalty = np.sum(self.episode_penalties[env_idx])
                        
                        # Store episode length
                        self.all_episode_lengths.append(ep_length)
                        self.moving_avg_length.append(ep_length)
                        self.zombie_penalties.append(total_ep_penalty)
                        
                        # Log episode completion with metrics
                        print(f"Episode {self.completed_episodes} completed - Env {env_idx}: Length = {ep_length} steps, Reward = {total_ep_reward:.2f}, Zombie penalties = {total_ep_penalty:.2f}")
                        
                        # Add to completed episodes and cumulative reward tracking
                        self.completed_episodes += 1
                        self.runs_cumulative_reward.append(total_ep_reward)
                        
                        # Log episode statistics to tensorboard
                        self.writer.add_scalar("episode/reward", total_ep_reward, self.completed_episodes)
                        self.writer.add_scalar("episode/length", ep_length, self.completed_episodes)
                        self.writer.add_scalar("episode/zombie_penalties", total_ep_penalty, self.completed_episodes)
                        
                        # Log moving averages every 10 episodes
                        if self.completed_episodes % 10 == 0 and len(self.moving_avg_length) > 0:
                            avg_length = np.mean(self.moving_avg_length)
                            self.writer.add_scalar("stats/avg_episode_length", avg_length, self.completed_episodes)
                            
                            if len(self.runs_cumulative_reward) > 0:
                                avg_reward = np.mean(self.runs_cumulative_reward[-min(len(self.runs_cumulative_reward), 100):])
                                self.writer.add_scalar("stats/avg_reward", avg_reward, self.completed_episodes)
                        
                        # Reset episode-specific tracking for this environment
                        self.episode_rewards[env_idx] = [0]
                        self.episode_lengths[env_idx] = [0]
                        self.episode_penalties[env_idx] = [0]
                
                # Log training progress periodically
                if self.global_step % self.log_freq == 0:
                    print(f"Step: {self.global_step}, Epsilon: {self.epsilon:.4f}, Episodes: {self.completed_episodes}, Curriculum: {curriculum_phase}")
                
                # # Sleep to avoid overwhelming the CPU
                # time.sleep(self.step_interval)
                
        except KeyboardInterrupt:
            print("Training interrupted")
            if self.should_save_model:
                # Save final model
                self.save_model("models/pvz_dqn_final.pt")
                # Log final summary
                self._log_final_summary()

    def save_model(self, path):
        """Save the Q-network to a file."""
        if self.q_network is None:
            print("No model to save")
            return
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.q_network.optimizer.state_dict(),
            'global_step': self.global_step,
            'epsilon': self.epsilon,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the Q-network from a file."""
        # Initialize networks if not done yet
        if self.q_network is None:
            self.initialize_networks()
            
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.q_network.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}")
    
    def close(self):
        """Close the connector and clean up resources."""
        self.connector.stop()
        self.writer.close()

    def _log_training_metrics(self):
        """Log training metrics focused on episode length to TensorBoard only."""
        # Calculate average episode length
        avg_length_all = np.mean(self.all_episode_lengths) if self.all_episode_lengths else 0
        avg_length_recent = np.mean(self.moving_avg_length) if self.moving_avg_length else 0
        
        # Calculate average reward
        avg_reward = np.mean(self.runs_cumulative_reward[-min(len(self.runs_cumulative_reward), 100):]) if self.runs_cumulative_reward else 0
        
        # Log to console only (no file output)
        print(f"\nEpisodes completed: {self.completed_episodes}")
        print(f"Average episode length (all): {avg_length_all:.2f}")
        print(f"Average episode length (last 100): {avg_length_recent:.2f}")
        
        # Log to tensorboard
        self.writer.add_scalar("training/episodes_completed", self.completed_episodes, self.global_step)
        self.writer.add_scalar("training/epsilon", self.epsilon, self.global_step)
        
        # Episode length stats
        if self.all_episode_lengths:
            self.writer.add_scalar("stats/avg_episode_length_all", avg_length_all, self.global_step)
        
        if self.moving_avg_length:
            self.writer.add_scalar("stats/avg_episode_length_100", avg_length_recent, self.global_step)
        
        # Loss stats (if available)
        if self.recent_losses:
            self.writer.add_scalar("losses/current", self.recent_losses[-1], self.global_step)
            self.writer.add_scalar("losses/avg_loss", np.mean(self.recent_losses), self.global_step)

    def _log_final_summary(self):
        """Log a final summary to console and create episode length histogram in TensorBoard."""
        # Calculate final statistics
        avg_length_all = np.mean(self.all_episode_lengths) if self.all_episode_lengths else 0
        avg_length_recent = np.mean(self.moving_avg_length) if self.moving_avg_length else 0
        
        # Calculate zombie penalty statistics
        avg_penalty_all = np.mean(self.zombie_penalties) if self.zombie_penalties else 0
        avg_penalty_recent = np.mean(list(self.zombie_penalties)[-100:]) if len(self.zombie_penalties) >= 100 else (np.mean(self.zombie_penalties) if self.zombie_penalties else 0)
        
        # Display final summary to console only
        print("\n===== TRAINING SUMMARY =====")
        print(f"Total steps: {self.global_step}")
        print(f"Episodes completed: {self.completed_episodes}")
        print(f"Average episode length (all episodes): {avg_length_all:.2f}")
        print(f"Average episode length (last 100): {avg_length_recent:.2f}")
        
        if self.all_episode_lengths:
            print(f"Shortest episode: {min(self.all_episode_lengths)}")
            print(f"Longest episode: {max(self.all_episode_lengths)}")
            
        # Print zombie penalty statistics
        print("\n----- Zombie Proximity Penalties -----")
        print(f"Average penalty per episode (all): {avg_penalty_all:.4f}")
        print(f"Average penalty per episode (last 100): {avg_penalty_recent:.4f}")
        
        if self.zombie_penalties:
            print(f"Max penalty in an episode: {min(self.zombie_penalties):.4f}")  # Most negative value
            print(f"Min penalty in an episode: {max(self.zombie_penalties):.4f}")  # Least negative value
        print("===========================\n")
        
        # Create a histogram of episode lengths in TensorBoard
        if len(self.all_episode_lengths) > 0:
            self.writer.add_histogram("final/episode_length_distribution", 
                                     np.array(self.all_episode_lengths), 
                                     self.global_step)
            
        # Create a histogram of zombie penalties in TensorBoard
        if len(self.zombie_penalties) > 0:
            self.writer.add_histogram("final/zombie_penalty_distribution", 
                                     np.array(self.zombie_penalties), 
                                     self.global_step)
            
        # Log final average statistics
        self.writer.add_scalar("final/avg_episode_length", avg_length_all, self.global_step)
        self.writer.add_scalar("final/avg_zombie_penalty", avg_penalty_all, self.global_step)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)



if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train a DQN agent for Plants vs Zombies')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of environments to run in parallel')
    parser.add_argument('--port', type=int, default=5555, help='Port for connecting to Java connector')
    parser.add_argument('--load_model', type=str, default=None, help='Path to a saved model to load')
    parser.add_argument('--no_fast_training', action='store_true', help='Disable fast training mode')
    args = parser.parse_args()
    
    # Create and start the agent
    agent = PvZDQNAgent(
        num_envs=args.num_envs, 
        port=args.port,
        fast_training=not args.no_fast_training
    )
    
    # Load model if specified
    if args.load_model:
        agent.load_model(args.load_model)
    
    # Register cleanup handler
    def cleanup():
        print("Cleaning up...")
        agent.close()
    
    atexit.register(cleanup)
    
    try:
        # Train the agent
        agent.run()
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        # Close the agent
        agent.close()