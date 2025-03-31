import os
import random
import time
from dataclasses import dataclass
import socket
import pickle
import numpy as np
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import deque

# Custom replay buffer for our environment
class ReplayBuffer:
    def __init__(self, buffer_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device
        
    def add(self, state, action, reward, next_state, done, action_mask, next_action_mask):
        self.buffer.append((state, action, reward, next_state, done, action_mask, next_action_mask))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones, action_masks, next_action_masks = zip(*batch)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(np.array(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_tensor = torch.FloatTensor(np.array(dones)).to(self.device)
        action_masks_tensor = torch.FloatTensor(np.array(action_masks)).to(self.device)
        next_action_masks_tensor = torch.FloatTensor(np.array(next_action_masks)).to(self.device)
        
        return (states_tensor, actions_tensor, rewards_tensor, 
                next_states_tensor, dones_tensor, 
                action_masks_tensor, next_action_masks_tensor)
    
    def __len__(self):
        return len(self.buffer)

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

class PvZDQNAgent:
    def __init__(self, 
                 action_space_size=182,
                 buffer_size=10000,
                 batch_size=64,
                 gamma=0.99,
                 learning_rate=1e-4,
                 target_update_freq=1000,
                 learning_starts=1000,
                 exploration_fraction=0.3,
                 start_epsilon=1.0,
                 end_epsilon=0.1,
                 host='localhost',
                 port=5555):
        
        # Environment parameters
        self.action_space_size = action_space_size
        self.host = host
        self.port = port
        
        # DQN parameters
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.target_update_freq = target_update_freq
        self.learning_starts = learning_starts
        
        # Exploration parameters
        self.exploration_fraction = exploration_fraction
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon = start_epsilon
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.q_network = None
        self.target_network = None
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, self.device)
        
        # Initialize optimizer
        self.optimizer = None
        
        # Initialize server
        self.server_socket = None
        self.client_socket = None
        self.running = False
        
        # Training stats
        self.global_step = 0
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_count = 0
        self.writer = SummaryWriter(f"runs/pvz_dqn_{int(time.time())}")
        
        # State tracking
        self.current_state = None
        self.current_action_mask = None
        
    def initialize_networks(self, input_shape):
        """Initialize networks once we know the input shape."""
        if self.q_network is None:
            self.q_network = PvZQNetwork(input_shape, self.action_space_size).to(self.device)
            self.target_network = PvZQNetwork(input_shape, self.action_space_size).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            print("Networks initialized")
    
    def start_server(self):
        """Start the socket server to listen for connections from the Java game."""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server started on {self.host}:{self.port}")
        self.running = True
        
        # Start the main loop in a separate thread
        threading.Thread(target=self.main_loop, daemon=True).start()
    
    def stop_server(self):
        """Stop the server and close all connections."""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        print("Server stopped")
        
        # Save the model
        if self.q_network:
            torch.save(self.q_network.state_dict(), f"models/pvz_dqn_{int(time.time())}.pt")
            print("Model saved")
    
    def main_loop(self):
        """Main server loop that accepts connections and processes game states."""
        while self.running:
            print("Waiting for connection...")
            try:
                self.client_socket, addr = self.server_socket.accept()
                print(f"Connected to {addr}")
                
                # Process data from this client until disconnection
                self.handle_client()
                
            except Exception as e:
                print(f"Error in main loop: {e}")
            finally:
                if self.client_socket:
                    self.client_socket.close()
                    self.client_socket = None
    
    def handle_client(self):
        """Handle communication with a connected client."""
        try:
            while self.running:
                # Receive data from Java
                data = self.receive_data()
                if not data:
                    print("Client disconnected")
                    break
                
                # Process the game state and select an action
                action_index = self.step(data)
                
                # Send the action back to Java
                self.send_action(action_index)
                
        except Exception as e:
            print(f"Error handling client: {e}")
    
    def receive_data(self):
        """Receive and deserialize data from the Java client."""
        try:
            # Receive the data
            data = pickle.loads(self.client_socket.recv(4096))
            return data
            
        except Exception as e:
            print(f"Error receiving data: {e}")
            return None
    
    def send_action(self, action_index):
        """Serialize and send an action index to the Java client."""
        try:
            # Convert action index to bytes
            action_bytes = action_index.to_bytes(4, byteorder='big')
            
            # Send the action
            self.client_socket.sendall(action_bytes)
            
        except Exception as e:
            print(f"Error sending action: {e}")
    
    def preprocess_state(self, data):
        """Convert the game state data to a format suitable for the neural network."""
        state = data.get('state', {})
        action_mask = np.array(data.get('actionMask', []))
        reward = data.get('reward', 0)
        
        # Extract tensor and game info
        tensor = np.array(state.get('tensor', [[[0]]])).astype(np.float32)
        game_info = np.array(state.get('gameInfo', [0])).astype(np.float32)
        
        # Normalize the data
        tensor = tensor / 100.0  # Assuming max health is around 100
        game_info = game_info / 100.0  # Assuming max sun score is around 100
        
        return tensor, game_info, action_mask, reward
    
    def step(self, data):
        """Process a step in the environment and return an action."""
        tensor, game_info, action_mask, reward = self.preprocess_state(data)
        
        # Initialize networks if not done yet
        if self.q_network is None:
            self.initialize_networks(tensor.shape)
        
        # Convert to PyTorch tensors
        tensor_tensor = torch.FloatTensor(tensor).unsqueeze(0).to(self.device)
        game_info_tensor = torch.FloatTensor(game_info).unsqueeze(0).to(self.device)
        action_mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        
        # Store the current state for the next step
        next_state = (tensor, game_info)
        next_action_mask = action_mask
        
        # If we have a previous state, add it to the replay buffer
        if self.current_state is not None:
            self.replay_buffer.add(
                self.current_state[0], self.current_action, reward, 
                next_state[0], False, self.current_action_mask, next_action_mask
            )
            
            # Update episode stats
            self.episode_reward += reward
            self.episode_length += 1
            
            # Train the network
            self.train()
        
        # Update current state
        self.current_state = next_state
        self.current_action_mask = next_action_mask
        
        # Select action (epsilon-greedy)
        self.epsilon = max(
            self.end_epsilon, 
            self.start_epsilon - self.global_step * (self.start_epsilon - self.end_epsilon) / 
            (self.exploration_fraction * 1000000)
        )
        
        if random.random() < self.epsilon:
            # Random action
            valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
            if not valid_actions:
                action = 0  # Default to DO_NOTHING
            else:
                action = random.choice(valid_actions)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.q_network(tensor_tensor, game_info_tensor, action_mask_tensor)
                action = torch.argmax(q_values, dim=1).item()
        
        # Store the action for the next step
        self.current_action = action
        
        # Update global step
        self.global_step += 1
        
        # Log stats
        if self.global_step % 100 == 0:
            self.writer.add_scalar("charts/epsilon", self.epsilon, self.global_step)
            self.writer.add_scalar("charts/buffer_size", len(self.replay_buffer), self.global_step)
            self.writer.add_scalar("charts/episode_reward", self.episode_reward, self.global_step)
            self.writer.add_scalar("charts/episode_length", self.episode_length, self.global_step)
            print(f"Step: {self.global_step}, Epsilon: {self.epsilon:.2f}, Reward: {self.episode_reward:.2f}")
        
        return action
    
    def train(self):
        """Train the Q-network using experiences from the replay buffer."""
        # Skip training if we don't have enough samples
        if len(self.replay_buffer) < self.learning_starts:
            return
        
        # Sample from replay buffer
        states, actions, rewards, next_states, dones, action_masks, next_action_masks = self.replay_buffer.sample(self.batch_size)
        
        # Convert states to tensor and game_info components
        tensor_states = states[:, :3]  # First 3 channels are the tensor
        game_info_states = states[:, 3:]  # Rest is game info
        
        tensor_next_states = next_states[:, :3]
        game_info_next_states = next_states[:, 3:]
        
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
        state_action_values = self.q_network(tensor_states, game_info_states).gather(1, actions.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_state_values = self.target_network(tensor_next_states, game_info_next_states, next_action_masks).max(1)[0]
            # Compute the expected Q values
            expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
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

if __name__ == "__main__":
    # Create directories for models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("runs", exist_ok=True)
    
    # Create and start the agent
    agent = PvZDQNAgent()
    try:
        agent.start_server()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        agent.stop_server() 