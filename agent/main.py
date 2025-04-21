import socket
import pickle
import numpy as np
import threading
import time
import random
import json
import torch
import argparse
import subprocess
import os
import signal
import sys
import atexit
import struct
from pvz_dqn import PvZDQNAgent


class GameConnector:
    """Manages communication with the Java game connector."""
    def __init__(self, host='localhost', port=5555):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        
    def start(self):
        """Start the connector and establish connection."""
        try:
            # Connect to the connector
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.running = True
            
            print(f"Connected to connector on {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Error connecting to connector: {e}")
            return False
    
    def create_envs(self, num_envs, max_steps=1000, training_mode=False):
        """Request the connector to create game environments.
        
        Args:
            num_envs: Number of environments to create
            max_steps: Maximum steps per episode
            training_mode: If True, runs in headless mode for faster training
                          If False, shows game window with agent actions
        """
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
                    'type': 'stop'
                }
                self.send_data(command)
            
            if self.socket:
                self.socket.close()
                self.socket = None
            
            self.running = False
            print("Connector stopped")
            return True
        except Exception as e:
            print(f"Error stopping connector: {e}")
            return False

class PvZAgent:
    def __init__(self, num_envs=1, port=5555, dqn_agent=None, model_path='pvz_dqn_step_11500.pt'):
        self.num_envs = num_envs
        self.connector = GameConnector(port=port)
        self.action_space_size = 182  # 4*5*9 + 2 (do nothing + collect sun)
        self.episode_rewards = []
        self.total_steps = 0
        self.dqn_agent = dqn_agent

        if self.dqn_agent is not None:
            self.dqn_agent.load_model(model_path)
        
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
    

    def select_action_dqn(self, state, action_mask):
        """Select an action using DQN agent."""
        return self.dqn_agent.select_action_render(state, action_mask)
    
    def select_action(self, state, action_mask):
        """Select an action using rule-based strategy."""
        # Convert state data to tensors
        tensor, game_info = self.preprocess_state(state)
        
        # Get valid actions
        valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
        
        if not valid_actions:
            return 0  # Return DO_NOTHING if no valid actions
        
        # Extract game info
        sun_score = game_info[0]
        
        # 1. Always collect sun if available
        if action_mask[1] == 1:  # ACTION_COLLECT_SUN
            return 1
        
        # 2. Plant sunflowers if we have enough sun
        if sun_score >= 50:
            # Try to plant sunflowers in the first two columns
            for lane in range(5):
                for col in range(2):
                    action_idx = 2 + (0 * 5 * 9) + (lane * 9) + col
                    if action_idx < len(action_mask) and action_mask[action_idx] == 1:
                        return action_idx
        
        # 3. Plant peashooters in lanes with zombies
        if sun_score >= 100:
            # Check which lanes have zombies
            zombie_lanes = []
            for lane in range(5):
                if np.sum(tensor[2][lane]) > 0:  # Check if there are zombies in this lane
                    zombie_lanes.append(lane)
            
            # Plant peashooters in lanes with zombies
            for lane in zombie_lanes:
                for col in range(2, 8):  # Columns 2-7
                    action_idx = 2 + (1 * 5 * 9) + (lane * 9) + col
                    if action_idx < len(action_mask) and action_mask[action_idx] == 1:
                        return action_idx
        
        # 4. If we have enough sun, plant freeze peashooters in lanes with zombies
        if sun_score >= 175:
            for lane in range(5):
                if np.sum(tensor[2][lane]) > 0:  # Check if there are zombies in this lane
                    for col in range(2, 8):  # Columns 2-7
                        action_idx = 2 + (2 * 5 * 9) + (lane * 9) + col
                        if action_idx < len(action_mask) and action_mask[action_idx] == 1:
                            return action_idx
        
        # 5. If we have enough sun, plant walnuts in front of our plants
        if sun_score >= 50:
            for lane in range(5):
                # Check if we have plants in this lane
                has_plants = False
                rightmost_plant_col = -1
                for col in range(9):
                    if tensor[0][lane][col] > 0:  # If there's a plant
                        has_plants = True
                        rightmost_plant_col = max(rightmost_plant_col, col)
                
                if has_plants and rightmost_plant_col < 8:
                    # Try to plant a walnut in front of our rightmost plant
                    action_idx = 2 + (3 * 5 * 9) + (lane * 9) + (rightmost_plant_col + 1)
                    if action_idx < len(action_mask) and action_mask[action_idx] == 1:
                        return action_idx
        
        # 6. If all else fails, choose a random valid action
        return random.choice(valid_actions)
    
    def run(self, max_steps=10000, training_mode=False):
        """Run the agent with rule-based strategy.
        
        Args:
            max_steps: Maximum number of steps to run
            training_mode: If True, runs in headless mode for faster training
                          If False, shows game window with agent actions
        """
        # Start the connector
        if not self.connector.start():
            print("Failed to start connector")
            return
        
        # Create environments
        if not self.connector.create_envs(self.num_envs, max_steps=1000, training_mode=training_mode):
            print("Failed to create environments")
            return
        
        # Reset all environments to get initial states
        states, action_masks = self.connector.reset()
        if states is None:
            print("Failed to reset environments")
            return
        
        episode_rewards = [0] * self.num_envs
        
        try:
            # Main loop
            for step in range(max_steps):
                # Select actions for all environments
                actions = [self.select_action(state, mask) if self.dqn_agent is None else self.select_action_dqn(state, mask)
                          for state, mask in zip(states, action_masks) ]
                
                # Execute actions in all environments
                next_states, next_action_masks, rewards, dones, truncateds, final_observations = self.connector.step(actions)
                
                if next_states is None:
                    print("Failed to step environments, resetting...")
                    states, action_masks = self.connector.reset()
                    if states is None:
                        print("Failed to reset environments")
                        break
                    continue
                
                # Update episode rewards
                for i, reward in enumerate(rewards):
                    episode_rewards[i] += reward
                
                # Handle episode terminations
                for i, (done, truncated) in enumerate(zip(dones, truncateds)):
                    if done or truncated:
                        print(f"Environment {i} finished episode with reward: {episode_rewards[i]}")
                        self.episode_rewards.append(episode_rewards[i])
                        episode_rewards[i] = 0
                
                # Update states for next iteration
                states = next_states
                action_masks = next_action_masks
                
                # Update total steps
                self.total_steps += self.num_envs
                
                # Print progress
                if step % 100 == 0:
                    avg_reward = np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                    print(f"Step: {step}, Total steps: {self.total_steps}, Avg reward (last 100): {avg_reward:.2f}")
                
                # Sleep to avoid overwhelming the CPU
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("Run interrupted")
        finally:
            # Stop the connector
            self.connector.stop()
    
    def close(self):
        """Close the connector."""
        self.connector.stop()

def launch_connector(port=5555):
    """Launch the Java connector process."""
    try:
        # Build the command
        cmd = ["java", "-jar", "../PlantsVsZombies.jar", "--connector", f"--port={port}"]
        
        print(f"Launching connector with command: {' '.join(cmd)}")
        
        # Launch the process
        process = subprocess.Popen(cmd)
        
        # Register cleanup handler
        def cleanup():
            if process.poll() is None:  # If process is still running
                print("Terminating connector process...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        atexit.register(cleanup)
        
        # Give the connector time to start
        time.sleep(2)
        
        return process
    except Exception as e:
        print(f"Error launching connector: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a rule-based PvZ agent')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of environments to run in parallel')
    parser.add_argument('--port', type=int, default=5555, help='Port for connecting to Java connector')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum number of steps to run')
    parser.add_argument('--training', action='store_true', help='Run in headless training mode')
    parser.add_argument('--model_path', type=str, default='./models/step_10000.pt',
                      help='Path to trained model (default: ./models/step_100000.pt)')

    
    args = parser.parse_args()
    
    # Launch Java connector
    connector_process = None

    connector_process = launch_connector(port=args.port) 
    if connector_process is None:
        print("Failed to launch connector")
        sys.exit(1)


    try:
        agent = PvZAgent(num_envs=args.num_envs, port=args.port, dqn_agent=PvZDQNAgent(num_envs=args.num_envs, port=args.port, model_path=args.model_path))
        print("Using DQN model")

    except Exception as e:
        agent = PvZAgent(num_envs=args.num_envs, port=args.port)
        print("Using rule-based strategy")

        
    # Register cleanup handler
    def cleanup():
        print("Cleaning up...")
        agent.close()
        if connector_process and connector_process.poll() is None:
            connector_process.terminate()
    
    atexit.register(cleanup)
    
    try:
        # Run with training mode flag
        agent.run(max_steps=args.max_steps, training_mode=args.training)
    except KeyboardInterrupt:
        print("Run interrupted")
    finally:
        # Close the agent
        agent.close()
