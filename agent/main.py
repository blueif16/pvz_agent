import socket
import pickle
import numpy as np
import threading
import time
import random
import json

class PvZAgent:
    def __init__(self, host='localhost', port=5555):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.action_space_size = 182  # 4*5*9 + 2 (do nothing + collect sun)
        
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
                action_index = self.select_action(data)
                
                # Send the action back to Java
                self.send_action(action_index)
                
                # Update cumulative reward
                self.current_episode_reward += data.get('reward', 0)
                
        except Exception as e:
            print(f"Error handling client: {e}")
    
    def receive_data(self):
        """Receive and deserialize data from the Java client."""
        try:
            # First receive the length of the data
            length_bytes = self.client_socket.recv(4)
            if not length_bytes:
                return None
            
            data_length = int.from_bytes(length_bytes, byteorder='big')
            
            # Then receive the actual data
            data_bytes = b''
            while len(data_bytes) < data_length:
                chunk = self.client_socket.recv(min(4096, data_length - len(data_bytes)))
                if not chunk:
                    return None
                data_bytes += chunk
            
            # Deserialize the data
            data = pickle.loads(data_bytes)
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
    
    def select_action(self, data):
        """Select an action based on the current game state."""
        # Extract data from the received package
        state = data.get('state', {})
        action_mask = data.get('actionMask', [])
        reward = data.get('reward', 0)
        
        # Convert tensor to numpy array for easier processing
        tensor = np.array(state.get('tensor', [[[0]]]))
        game_info = np.array(state.get('gameInfo', [0]))
        
        # Get valid actions (where mask is 1)
        valid_actions = [i for i, mask in enumerate(action_mask) if mask == 1]
        
        if not valid_actions:
            return 0  # Return DO_NOTHING if no valid actions
        
        # For now, implement a simple rule-based strategy
        # 1. Collect sun if available
        if action_mask[1] == 1:  # ACTION_COLLECT_SUN
            return 1
        
        # 2. Plant sunflowers in the back columns if we have enough sun
        sun_score = game_info[0]
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

    def end_episode(self):
        """End the current episode and record statistics."""
        self.episode_rewards.append(self.current_episode_reward)
        print(f"Episode ended with reward: {self.current_episode_reward}")
        self.current_episode_reward = 0

if __name__ == "__main__":
    agent = PvZAgent()
    try:
        agent.start_server()
        
        # Keep the main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        agent.stop_server()
