package core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.net.Socket;
import java.net.ServerSocket;
import java.util.Arrays;
import java.io.ByteArrayOutputStream;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.io.ByteArrayInputStream;
import java.io.PrintWriter;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import entities.plants.Plant;
import entities.zombies.Zombie;

import core.StateAction;
import core.StepInfo;

import java.io.OutputStream;
import java.io.InputStream;

public class Connector {
    
    private static final String SERVER_ADDRESS = "localhost";
    private static int SERVER_PORT = 5555;
    private static ServerSocket serverSocket;
    private static Socket clientSocket;
    private static OutputStream outputStream;
    private static InputStream inputStream;
    private static boolean connected = false;
    private static boolean closed = false;
    private static boolean running = false;
    
    // Game instances
    private static Map<Integer, Game> gameInstances = new ConcurrentHashMap<>();
    
    // Initialize server to listen for Python agent
    public static boolean startServer() {
        try {
            serverSocket = new ServerSocket(SERVER_PORT);
            System.out.println("Connector started on port " + SERVER_PORT);
            
            // Wait for client connection
            System.out.println("Waiting for client connection...");
            clientSocket = serverSocket.accept();
            System.out.println("Client connected: " + clientSocket.getInetAddress());
            
            // Set up input/output streams
            outputStream = clientSocket.getOutputStream();
            inputStream = clientSocket.getInputStream();
            
            connected = true;
            running = true;
            
            // Start command handling loop
            handleCommands();
            return true;
        } catch (IOException e) {
            System.err.println("Error starting connector: " + e.getMessage());
            e.printStackTrace();
            closeConnection();
            return false;
        }
    }
    
    // Listen for commands from the Python agent
    private static void listenForCommands() {
        try {
            System.out.println("Waiting for agent connection...");
            clientSocket = serverSocket.accept();
            System.out.println("Agent connected from " + clientSocket.getInetAddress());
            
            outputStream = clientSocket.getOutputStream();
            inputStream = clientSocket.getInputStream();
            connected = true;
            
            while (running && connected) {
                try {
                    // Receive command from agent
                    Map<String, Object> command = receiveCommand();
                    
                    if (command != null) {
                        // Process command
                        processCommand(command);
                    }
                } catch (Exception e) {
                    System.err.println("Error processing command: " + e.getMessage());
                    connected = false;
                }
            }
        } catch (IOException e) {
            System.err.println("Server error: " + e.getMessage());
        } finally {
            closeConnection();
        }
    }
    
    // Process commands from the agent
    private static void processCommand(Map<String, Object> command) {
        String type = (String) command.get("type");
        
        try {
            switch (type) {
                case "create_envs":
                    handleCreateEnvs(command);
                    break;
                case "reset":
                    handleReset(command);
                    break;
                case "step":
                    handleStep(command);
                    break;
                case "stop":
                    handleStop();
                    break;
                case "close":
                    handleClose();
                    break;
                default:
                    System.err.println("Unknown command type: " + type);
                    sendErrorResponse("Unknown command type: " + type);
            }
        } catch (Exception e) {
            System.err.println("Error processing command: " + e.getMessage());
            e.printStackTrace();
            sendErrorResponse("Error processing command: " + e.getMessage());
        }
    }
    
    // Handle create_envs command
    private static void handleCreateEnvs(Map<String, Object> command) {
        int numEnvs = ((Number) command.get("num_envs")).intValue();
        int maxSteps = ((Number) command.get("max_steps")).intValue();
        boolean training_mode = command.containsKey("training_mode") ? (Boolean) command.get("training_mode") : true;
        
        System.out.println("Creating " + numEnvs + " game environments (training_mode=" + training_mode + ")");
        
        // Close any existing game instances
        for (Game game : gameInstances.values()) {
            game.stop();
        }
        gameInstances.clear();
        
        // Create new game instances
        for (int i = 0; i < numEnvs; i++) {
            Game game = new Game(maxSteps, training_mode);
            if (!training_mode) {
                game.setAgentMode(true);
            }
            // if (training_mode) {
            //     // If in training mode, don't create the window
            //     game.setHeadless(true);
            // } else {
            //     // In agent mode, set agent mode flag but keep the window
            //     game.setAgentMode(true);
            // }
            gameInstances.put(i, game);
            game.start();
        }
        
        // Send success response
        Map<String, Object> response = new HashMap<>();
        response.put("status", "success");
        response.put("message", "Created " + numEnvs + " environments");
        sendResponse(response);
    }
    
    // Handle reset command - now resets the single game instance
    private static void handleReset(Map<String, Object> command) {
        // We'll use the first game instance (index 0)
        if (gameInstances.isEmpty()) {
            sendErrorResponse("No game environments exist");
            return;
        }
        
        List<float[]> states = new ArrayList<>();
        List<int[]> actionMasks = new ArrayList<>();

        for (Game game : gameInstances.values()) {
            StateAction stateAction = game.reset();
            states.add(stateAction.getState());
            actionMasks.add(stateAction.getAction());
        }
        
        // Send response with initial state
        Map<String, Object> response = new HashMap<>();
        response.put("status", "success");
        response.put("states", states);
        response.put("action_masks", actionMasks);
        sendResponse(response);
    }
    
    // Handle step command - steps multiple game instances with corresponding actions
    private static void handleStep(Map<String, Object> command) {
        // Extract the actions array
        @SuppressWarnings("unchecked")
        List<Object> actionsList = (List<Object>) command.get("actions");
        
        // Convert Double values to Integer
        int[] actions = new int[actionsList.size()];
        for (int i = 0; i < actionsList.size(); i++) {
            // Handle both Integer and Double cases
            Object actionObj = actionsList.get(i);
            if (actionObj instanceof Double) {
                actions[i] = ((Double) actionObj).intValue();
            } else if (actionObj instanceof Integer) {
                actions[i] = (Integer) actionObj;
            } else {
                actions[i] = Integer.parseInt(actionObj.toString());
            }
        }
        
        // Check if we have enough game instances
        if (gameInstances.isEmpty()) {
            sendErrorResponse("No game environments exist");
            return;
        }
        
        if (actions.length > gameInstances.size()) {
            sendErrorResponse("More actions provided than available environments");
            return;
        }
        
        // Lists to collect results from all environments
        List<float[]> states = new ArrayList<>();
        List<int[]> actionMasks = new ArrayList<>();
        List<Double> rewards = new ArrayList<>();
        List<Boolean> dones = new ArrayList<>();
        List<Boolean> truncateds = new ArrayList<>();
        List<float[]> finalObservations = new ArrayList<>();
        
        // Step each environment with its corresponding action
        for (int i = 0; i < actions.length; i++) {
            Game game = gameInstances.get(i);
            int action = actions[i];
            
            // Execute step in this environment
            StepInfo stepInfo = game.step(action);
            
            // Add other results to their respective lists
            states.add(stepInfo.newState);
            actionMasks.add(stepInfo.newAction);
            rewards.add(stepInfo.reward);
            dones.add(stepInfo.done);
            truncateds.add(stepInfo.truncated);
            
            // Add final observation if available
            if (stepInfo.final_obs != null) {
                finalObservations.add(stepInfo.final_obs);
            } else {
                finalObservations.add(null);
            }
        }
        
        // Send response with all results
        Map<String, Object> response = new HashMap<>();
        response.put("status", "success");
        response.put("states", states);
        response.put("action_masks", actionMasks);
        response.put("rewards", rewards);
        response.put("dones", dones);
        response.put("truncateds", truncateds);
        response.put("final_observations", finalObservations);
        sendResponse(response);
    }
    
    // Handle stop command
    private static void handleStop() {
        // Close all game instances
        for (Game game : gameInstances.values()) {
            game.stop();
        }
        gameInstances.clear();
        
        // Send success response
        Map<String, Object> response = new HashMap<>();
        response.put("status", "success");
        response.put("message", "Stopped all environments");
        sendResponse(response);
        
        // Close connection
        closeConnection();
    }
    
    // Handle close command
    private static void handleClose() {
        // Send success response
        Map<String, Object> response = new HashMap<>();
        response.put("status", "success");
        response.put("message", "Closing connection");
        sendResponse(response);
        
        // Close connection
        closeConnection();
    }
    
    // Send response to agent
    private static void sendResponse(Map<String, Object> response) {
        try {
            // Convert to JSON
            Gson gson = new GsonBuilder().create();
            String jsonResponse = gson.toJson(response);
            
            // Send the data length first (as 4 bytes)
            int dataLength = jsonResponse.getBytes("UTF-8").length;
            byte[] lengthBytes = new byte[4];
            lengthBytes[0] = (byte) ((dataLength >> 24) & 0xFF);
            lengthBytes[1] = (byte) ((dataLength >> 16) & 0xFF);
            lengthBytes[2] = (byte) ((dataLength >> 8) & 0xFF);
            lengthBytes[3] = (byte) (dataLength & 0xFF);
            
            outputStream.write(lengthBytes);
            
            // Then send the actual JSON data
            outputStream.write(jsonResponse.getBytes("UTF-8"));
            outputStream.flush();
        } catch (IOException e) {
            System.err.println("Error sending response: " + e.getMessage());
            closeConnection();
        }
    }
    
    // Send error response
    private static void sendErrorResponse(String errorMessage) {
        Map<String, Object> response = new HashMap<>();
        response.put("status", "error");
        response.put("message", errorMessage);
        sendResponse(response);
    }
    
    // Close connection
    public static boolean closeConnection() {
        if (!connected) return false;
        
        try {
            if (outputStream != null) outputStream.close();
            if (inputStream != null) inputStream.close();
            if (clientSocket != null) clientSocket.close();
            if (serverSocket != null) serverSocket.close();
            
            // Close all game instances
            for (Game game : gameInstances.values()) {
                game.stop();
            }
            gameInstances.clear();
            
            connected = false;
            running = false;
            closed = true;
            System.out.println("Disconnected from agent");
        } catch (IOException e) {
            System.err.println("Error closing connection: " + e.getMessage());
            closed = false; 
        }

        return closed;
    }
    
    
    
    // Set port for the connector
    public static void setPort(int port) {
        SERVER_PORT = port;
    }
    
    // Main method to start the connector as a standalone process
    public static void main(String[] args) {
        // Parse command line arguments
        for (String arg : args) {
            if (arg.startsWith("--port=")) {
                String portStr = arg.substring(7);
                try {
                    int port = Integer.parseInt(portStr);
                    setPort(port);
                    System.out.println("Using port: " + port);
                } catch (NumberFormatException e) {
                    System.err.println("Invalid port number: " + portStr);
                }
            }
        }
        
        // Start the connector server
        if (startServer()) {
            System.out.println("Connector server started successfully");
            
            // Add shutdown hook to clean up resources
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                System.out.println("Shutting down connector server...");
                closeConnection();
            }));
            
            // Keep the main thread alive
            while (running) {
                try {
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        } else {
            System.err.println("Failed to start connector server");
        }
    }

    // Receive command from the client
    private static Map<String, Object> receiveCommand() {
        try {
            // First receive the data length (4 bytes)
            byte[] lengthBytes = new byte[4];
            int bytesRead = inputStream.read(lengthBytes);
            
            if (bytesRead != 4) {
                return null;
            }
            
            int dataLength = ((lengthBytes[0] & 0xFF) << 24) |
                             ((lengthBytes[1] & 0xFF) << 16) |
                             ((lengthBytes[2] & 0xFF) << 8) |
                             (lengthBytes[3] & 0xFF);
            
            // Then receive the actual data
            byte[] dataBytes = new byte[dataLength];
            int totalBytesRead = 0;
            
            while (totalBytesRead < dataLength) {
                int bytesRemaining = dataLength - totalBytesRead;
                int bytesReadThisTime = inputStream.read(dataBytes, totalBytesRead, bytesRemaining);
                
                if (bytesReadThisTime == -1) {
                    return null;
                }
                
                totalBytesRead += bytesReadThisTime;
            }
            
            // Parse JSON data
            String jsonString = new String(dataBytes, "UTF-8");
            Gson gson = new GsonBuilder().create();
            return gson.fromJson(jsonString, Map.class);
        } catch (IOException e) {
            System.err.println("Error receiving command: " + e.getMessage());
            closeConnection();
            return null;
        }
    }

    private static void handleCommands() {
        System.out.println("Starting command handling loop");
        while (running) {
            try {
                Map<String, Object> command = receiveCommand();
                if (command == null) {
                    System.out.println("Received null command, closing connection");
                    break;
                }
                
                String type = (String) command.get("type");
                // System.out.println("Received command: " + type);
                
                switch (type) {
                    case "create_envs":
                        handleCreateEnvs(command);
                        break;
                    case "reset":
                        handleReset(command);
                        break;
                    case "step":
                        handleStep(command);
                        break;
                    case "close":
                        handleClose();
                        break;
                    default:
                        System.out.println("Unknown command: " + type);
                        Map<String, Object> response = new HashMap<>();
                        response.put("status", "error");
                        response.put("message", "Unknown command: " + type);
                        sendResponse(response);
                        break;
                }
            } catch (Exception e) {
                System.err.println("Error handling command: " + e.getMessage());
                e.printStackTrace();
                break;
            }
        }
        
        closeConnection();
    }
}
