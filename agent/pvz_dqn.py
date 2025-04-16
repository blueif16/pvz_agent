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
from collections import deque
import subprocess
import atexit
import signal
import sys
from replayBuffer import ReplayBuffer
import gymnasium as gym

# Neural network for DQN
class PvZQNetwork(nn.Module):
    def __init__(self, input_shape, action_space_size):
        super().__init__()
        
        # Process the 3D tensor (3x5x9) with convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate the size after flattening
        conv_output_size = 32 * 5 * 9
        
        # Process the game info vector
        self.game_info_layers = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU()
        )
        
        # Combine both processed inputs
        self.combined_layers = nn.Sequential(
            nn.Linear(conv_output_size + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_size)
        )
        
    def forward(self, tensor_input, game_info_input, action_mask=None):
        # Process the tensor input
        tensor_features = self.conv_layers(tensor_input)
        
        # Process the game info input
        game_info_features = self.game_info_layers(game_info_input)
        
        # Combine features
        combined = torch.cat([tensor_features, game_info_features], dim=1)
        
        # Get Q-values
        q_values = self.combined_layers(combined)
        
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
    
    def create_envs(self, num_envs, max_steps=1000, training_mode=True):
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
        """Send data to the connector."""
        try:
            # Convert numpy arrays to lists
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
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

class PvZDQNAgent:
    """DQN agent for Plants vs Zombies."""
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
        self.buffer_size = 10000
        self.batch_size = 128
        self.gamma = 0.99
        self.learning_rate = 2.5e-4
        self.epsilon = 1.0
        self.epsilon_start = 1.0
        self.epsilon_final = 0.05
        self.epsilon_decay = 50000
        self.target_update_freq = 500
        self.global_step = 0

        # Training parameters(needs to be tuned)
        self.learning_starts = 1000
        self.train_frequency = 4
        self.should_save_model = True
        
        # Initialize replay buffer
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3*5*9 + 6,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.action_space_size)
        
        self.replay_buffer = ReplayBuffer(
            buffer_size=self.buffer_size,
            observation_space=self.observation_space,
            action_space=self.action_space,
            device=self.device,
            handle_timeout_termination=False,
        )
        
        # Initialize networks
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        
        # Initialize tracking variables
        self.episode_rewards = []
        self.writer = SummaryWriter(f"runs/pvz_dqn_{int(time.time())}")
        self.reward_buffer = deque(maxlen=500)
        self.total_steps = 0
    
    def initialize_networks(self, input_shape=(3, 5, 9)):
        """Initialize the Q-network and target network."""
        action_space_size = 182  # 4*5*9 + 2 (do nothing + collect sun)
        
        self.q_network = PvZQNetwork(input_shape, action_space_size).to(self.device)
        self.target_network = PvZQNetwork(input_shape, action_space_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        print("Networks initialized")
    
    def preprocess_state(self, state):
        """
        Unpack the flattened state from Java into tensor and game info components.
        
        Args:
            state: 1D array containing flattened tensor (3x5x9) and game info (6 values)
            
        Returns:
            tensor: 3D tensor of shape (3, 5, 9)
            game_info: 1D array of length 6
        """
        if state is None:
            # Return zeros if state is None (error case)
            return np.zeros((3, 5, 9)), np.zeros(6)
        
        # Convert to numpy array if it's not already
        state_array = np.array(state, dtype=np.float32)
        
        # Extract tensor data (first 3*5*9 = 135 elements)
        tensor_flat = state_array[:135]
        tensor = tensor_flat.reshape(3, 5, 9)
        
        # Extract game info (last 6 elements)
        game_info = state_array[135:]
        
        return tensor, game_info
    
    def select_action_render(self, state, action_mask):
        """Select an action using epsilon-greedy policy."""
        # Convert state data to tensors
        tensor, game_info = self.preprocess_state(state)
        
        # Get valid actions
        valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
        
        if not valid_actions:
            return 0  # Return DO_NOTHING if no valid actions
        

        tensor_tensor = torch.FloatTensor(tensor).unsqueeze(0).to(self.device)
        game_info_tensor = torch.FloatTensor(game_info).unsqueeze(0).to(self.device)
        action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        
        q_values = self.q_network(tensor_tensor, game_info_tensor, action_mask_tensor)
        
        return torch.argmax(q_values, dim=1).item()
    
    def select_action(self, state, action_mask):
        """Select an action using epsilon-greedy policy."""
        # Convert state data to tensors
        tensor, game_info = self.preprocess_state(state)
        
        # Get valid actions
        valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
        
        if not valid_actions:
            return 0  # Return DO_NOTHING if no valid actions
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Random action
            return random.choice(valid_actions)
        else:
            # Greedy action
            with torch.no_grad():
                tensor_tensor = torch.FloatTensor(tensor).unsqueeze(0).to(self.device)
                game_info_tensor = torch.FloatTensor(game_info).unsqueeze(0).to(self.device)
                action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
                
                q_values = self.q_network(tensor_tensor, game_info_tensor, action_mask_tensor)
                
                return torch.argmax(q_values, dim=1).item()
            

    
    
    def _update_network(self):
        """Update the Q-network using experiences from the replay buffer."""
        # Sample from replay buffer
        # print("sampling from replay buffer and updating network")
        data = self.replay_buffer.sample(self.batch_size)
        
        # Get batch data
        states = data.observations
        actions = data.actions
        rewards = data.rewards.flatten()
        next_states = data.next_observations
        dones = data.dones.flatten()
        next_action_masks = data.next_action_masks
    
        # Process states and next_states
        batch_size = states.shape[0]
        
        # Extract tensor and game_info from states
        tensor_states = states[:, :135].reshape(batch_size, 3, 5, 9)
        game_info_states = states[:, 135:]
        
        tensor_next_states = next_states[:, :135].reshape(batch_size, 3, 5, 9)
        game_info_next_states = next_states[:, 135:]
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        q_values = self.q_network(tensor_states, game_info_states)
        state_action_values = q_values.gather(1, actions).squeeze()
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_q_values = self.target_network(tensor_next_states, game_info_next_states, next_action_masks)
            # Take max over next Q-values (standard DQN approach)
            next_state_values = next_q_values.max(1)[0]
            # Compute the expected Q values
            expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        if self.global_step % 1000 == 0:
            print("loss: ", loss.item())
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        for param in self.q_network.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update target network
        if self.global_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            print("Target network updated")
        
        # Log loss
        if self.global_step % 100 == 0:
            self.writer.add_scalar("losses/q_loss", loss.item(), self.global_step)
            self.writer.add_scalar("training/epsilon", self.epsilon, self.global_step)
    
    def run(self, max_steps=100000, training_mode=True):
        """Start training with multiple environments."""
        # Start the connector
        if not self.connector.start():
            print("Failed to start connector")
            return
        
        # Create environments
        if not self.connector.create_envs(self.num_envs, max_steps=1000, training_mode=training_mode):
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

        try:
            # Training loop
            while self.total_steps < max_steps:
                # Select actions for all environments
                actions = [self.select_action(state, mask) 
                          for state, mask in zip(states, action_masks)]
                
                # Execute actions in all environments
                next_states, next_action_masks, rewards, dones, truncateds, final_observations = self.connector.step(actions)
                
                if next_states is None:
                    print("Failed to step environments, resetting...")
                    states, action_masks = self.connector.reset()
                    if states is None:
                        print("Failed to reset environments")
                        break
                    continue
                
                # Process each environment's results
                real_next_states = next_states.copy()
                for i in range(self.num_envs):
                    if truncateds[i]:
                        real_next_states[i] = final_observations[i]

                # print("actions: ", len(actions))
                    
                # Convert to numpy arrays and add to replay buffer
                self.replay_buffer.add(
                    np.array(states),
                    np.array(real_next_states),
                    np.array(actions),
                    np.array(rewards),
                    np.array(dones),
                    [{"next_action_masks": mask} for mask in next_action_masks]
                )
                
                states = next_states
                action_masks = next_action_masks

                # Update global step count
                self.global_step += self.num_envs
                
                # Update epsilon using linear schedule
                self.epsilon = linear_schedule(
                    self.epsilon_start, 
                    self.epsilon_final,
                    self.epsilon_decay,
                    self.global_step
                )
                
                # Train the network if we have enough samples
                if self.global_step > self.learning_starts and self.global_step % self.train_frequency == 0:
                    self._update_network()
                
                # Update target network
                if self.global_step % self.target_update_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                    print("Target network updated")
                    if self.should_save_model:
                        self.save_model(f"models/pvz_dqn_step{self.global_step}.pt")
                
                # Log training progress
                if self.global_step % 100 == 0:
                    print(f"Step: {self.global_step}, Epsilon: {self.epsilon:.4f}")
                
                # Store rewards
                for env_idx in range(self.num_envs):
                    self.reward_buffer.append(rewards[env_idx])
                    self.total_steps += 1
                    
                    # Print statistics every 500 steps
                    if self.total_steps % 500 == 0:
                        avg_reward = np.mean(self.reward_buffer)
                        print(f"Step {self.total_steps}: Avg reward (last 500) = {avg_reward:.2f}")
                    
                # Episode tracking
                for done in dones:
                    if done:
                        if len(self.episode_rewards) > 0:
                            avg_ep_reward = np.mean(self.episode_rewards[-100:])
                            print(f"Episode {len(self.episode_rewards)}: Avg reward (last 100) = {avg_ep_reward:.2f}")
                        self.episode_rewards.append(np.sum(rewards))
                        rewards = []
                
                # Sleep to avoid overwhelming the CPU
                time.sleep(0.001)
                
        except KeyboardInterrupt:
            print("Training interrupted")
            if self.should_save_model:
                self.save_model("models/pvz_dqn_final.pt")
    
    def save_model(self, path):
        """Save the Q-network to a file."""
        if self.q_network is None:
            print("No model to save")
            return
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epsilon': self.epsilon,
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load the Q-network from a file."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Initialize networks if not done yet
        if self.q_network is None:
            # Assume standard input shape
            self.initialize_networks((3, 5, 9))
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {path}")
    
    def close(self):
        """Close the connector and clean up resources."""
        self.connector.stop()
        self.writer.close()

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