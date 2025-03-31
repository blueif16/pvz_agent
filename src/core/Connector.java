package core;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.ObjectInputStream;
import java.net.Socket;
import java.util.Arrays;

import entities.plants.Plant;
import entities.zombies.Zombie;

public class Connector {
    
    private static final String SERVER_ADDRESS = "localhost";
    private static final int SERVER_PORT = 5555;
    private static Socket socket;
    private static ObjectOutputStream outputStream;
    private static ObjectInputStream inputStream;
    private static boolean connected = false;
    
    // Initialize connection to Python backend
    public static void initConnection() {
        try {
            socket = new Socket(SERVER_ADDRESS, SERVER_PORT);
            outputStream = new ObjectOutputStream(socket.getOutputStream());
            inputStream = new ObjectInputStream(socket.getInputStream());
            connected = true;
            System.out.println("Connected to Python backend at " + SERVER_ADDRESS + ":" + SERVER_PORT);
        } catch (IOException e) {
            System.err.println("Failed to connect to Python backend: " + e.getMessage());
            connected = false;
        }
    }
    
    // Close connection
    public static void closeConnection() {
        if (!connected) return;
        
        try {
            if (outputStream != null) outputStream.close();
            if (inputStream != null) inputStream.close();
            if (socket != null) socket.close();
            connected = false;
            System.out.println("Disconnected from Python backend");
        } catch (IOException e) {
            System.err.println("Error closing connection: " + e.getMessage());
        }
    }
    
    public static int send(GameState gameState, int[] actionMask, int reward) {
        if (!connected) {
            initConnection();
            if (!connected) {
                return fallbackStrategy(actionMask);
            }
        }
        
        try {
            Map<String, Object> stateData = processGameState(gameState);
            
            Map<String, Object> dataPackage = new HashMap<>();
            dataPackage.put("state", stateData);
            dataPackage.put("reward", reward);
            dataPackage.put("actionMask", actionMask);
            
            outputStream.writeObject(dataPackage);
            outputStream.flush();
            
            Integer actionIndex = (Integer) inputStream.readObject();
            
            if (actionIndex >= 0 && actionIndex < actionMask.length && actionMask[actionIndex] == 1) {
                return actionIndex;
            } else {
                System.err.println("Received invalid action index: " + actionIndex);
                return fallbackStrategy(actionMask);
            }
            
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error communicating with Python backend: " + e.getMessage());
            connected = false;
            return fallbackStrategy(actionMask);
        }
    }
    
    private static Map<String, Object> processGameState(GameState gameState) {
        Map<String, Object> state = new HashMap<>();
        
        int[][][] tensor = new int[3][5][9];

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 5; j++) {
                Arrays.fill(tensor[i][j], 0);
            }
        }

        for (int lane = 0; lane < 5; lane++) {
            for (Plant plant : gameState.getPlantsInLane(lane)) {
                int col = plant.getGridX();
                if (col >= 0 && col < 9) {
                    tensor[0][lane][col] = getPlantTypeId(plant);
                    tensor[1][lane][col] = plant.getHealth();
                }
            }
        }
        
        for (int lane = 0; lane < 5; lane++) {
            for (Zombie zombie : gameState.getZombiesInLane(lane)) {
                int col = zombie.getColumn();
                if (col >= 0 && col < 9) {
                    tensor[2][lane][col] += zombie.getHealth();
                }
            }
        }
        
        int[] gameInfo = new int[6];
        
        gameInfo[0] = gameState.getSunScore();
        gameInfo[1] = gameState.getSuns().size();

        for (int i = 2; i < 6; i++) {
            gameInfo[i] = gameState.getCardCooldown(i - 2);
        }
        
        state.put("tensor", tensor);
        state.put("gameInfo", gameInfo);
        
        return state;
    }
    
    private static int getPlantTypeId(Plant plant) {
        String className = plant.getName();
        switch (className) {
            case "Sunflower": return 1;
            case "Peashooter": return 2;
            case "FreezePeashooter": return 3;
            case "Walnut": return 4;
            default: return 0;
        }
    }
    
    // Fallback strategy if connection fails
    private static int fallbackStrategy(int[] actionMask) {
        return RewardCounter.ACTION_DO_NOTHING;
    }
}
