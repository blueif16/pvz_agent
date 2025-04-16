package core;

import entities.zombies.Zombie;
import java.util.ArrayList;
import entities.plants.Plant;

public class RewardCounter {
    private GameState gameState;
    private double reward;
    
    // Total number of actions: 4 plant types * 5 lanes * 9 columns + 1 (collect sun) + 1 (do nothing)
    public static final int ACTION_SPACE_SIZE = 4 * 5 * 9 + 2;
    
    // 1,2 action manually set, planting action start at index 2
    public static final int ACTION_DO_NOTHING = 0;
    public static final int ACTION_COLLECT_SUN = 1;

    
    public RewardCounter(GameState gs) {
        gameState = gs;
    }
    
    public void collectSun() {
        if (!gameState.getSuns().isEmpty()) {
            gameState.getSuns().get(0).collect();
        }
    }
    

    public int[] getActionMask() {
        int[] actionMask = new int[ACTION_SPACE_SIZE];
        
 
        actionMask[ACTION_DO_NOTHING] = 1;
        

        actionMask[ACTION_COLLECT_SUN] = gameState.getSuns().isEmpty() ? 0 : 1;
        

        boolean[][] occupied = new boolean[5][9];
        

        for (int lane = 0; lane < 5; lane++) {
            for (Plant plant : gameState.getPlantsInLane(lane)) {
                int col = plant.getGridX();
                if (col >= 0 && col < 9) {
                    occupied[lane][col] = true;
                }
            }
        }
        

        boolean[] plantTypeAvailable = new boolean[4];
        for (int i = 0; i < 4; i++) {
            plantTypeAvailable[i] = gameState.getCardCooldown(i) == 0;
        }
        

        int sunScore = gameState.getSunScore();
        boolean canAffordSunflower = sunScore >= 50;
        boolean canAffordPeashooter = sunScore >= 100;
        boolean canAffordFreezePeashooter = sunScore >= 175;
        boolean canAffordWallnut = sunScore >= 50;
        

        int actionIndex = 2; 

        for (int plantType = 0; plantType < 4; plantType++) {
            boolean canAfford = false;
            switch (plantType) {
                case 0: canAfford = canAffordSunflower; break;
                case 1: canAfford = canAffordPeashooter; break;
                case 2: canAfford = canAffordFreezePeashooter; break;
                case 3: canAfford = canAffordWallnut; break;
            }

            for (int lane = 0; lane < 5; lane++) {
                for (int col = 0; col < 9; col++) {
          
                    if (!occupied[lane][col] && plantTypeAvailable[plantType] && canAfford) {
                        actionMask[actionIndex] = 1;
                    } else {
                        actionMask[actionIndex] = 0;
                    }
                    actionIndex++;
                }
            }
        }
        
        return actionMask;
    }
    
    //Executes the action corresponding to the given action index.
    
    public void executeAction(int actionIndex, Game game) {
        if (actionIndex == ACTION_DO_NOTHING) {
            return;
        }
        
        if (actionIndex == ACTION_COLLECT_SUN) {
            collectSun();
            return;
        }

        int adjustedIndex = actionIndex - 2; 
        int plantType = adjustedIndex / (5 * 9); 
        int remainingIndex = adjustedIndex % (5 * 9);
        int lane = remainingIndex / 9;
        int col = remainingIndex % 9;
        
        // Check if the action is valid before executing
        int[] actionMask = getActionMask();
        if (actionMask[actionIndex] == 0) {
            // Invalid action, do nothing
            return;
        }
        
        // Select the appropriate plant type
        switch (plantType) {
            case 0: // Sunflower
                game.getWindow().selectPlant(entities.plants.Sunflower.class, 0);
                break;
            case 1: // Peashooter
                game.getWindow().selectPlant(entities.plants.Peashooter.class, 1);
                break;
            case 2: // FreezePeashooter
                game.getWindow().selectPlant(entities.plants.FreezePeashooter.class, 2);
                break;
            case 3: // Wallnut
                game.getWindow().selectPlant(entities.plants.Walnut.class, 3);
                break;
        }
        
        // Plant at the specified location
        // The InputHandler.plantSelected method will handle setting the cooldown
        game.getInputHandler().plantSelected(col, lane);
    }
    
    
    public static int getActionSpaceSize() {
        return ACTION_SPACE_SIZE;
    }

    public static float[] processGameState(GameState gameState) {
        // Create a 1D array to hold all state information
        // Format: [tensor (3x5x9 flattened), gameInfo (6 values)]
        float[] state = new float[3*5*9 + 6];
        
        // Fill tensor data (flattened 3D tensor)
        int index = 0;
        
        // Layer 0: Plant types
        for (int lane = 0; lane < 5; lane++) {
            for (int col = 0; col < 9; col++) {
                int plantType = 0;
                for (Plant plant : gameState.getPlantsInLane(lane)) {
                    if (plant.getGridX() == col) {
                        plantType = getPlantTypeId(plant);
                        break;
                    }
                }
                state[index++] = plantType;
            }
        }
        
        // Layer 1: Plant health
        for (int lane = 0; lane < 5; lane++) {
            for (int col = 0; col < 9; col++) {
                int plantHealth = 0;
                for (Plant plant : gameState.getPlantsInLane(lane)) {
                    if (plant.getGridX() == col) {
                        plantHealth = plant.getHealth();
                        break;
                    }
                }
                state[index++] = plantHealth;
            }
        }
        
        // Layer 2: Zombie health
        for (int lane = 0; lane < 5; lane++) {
            for (int col = 0; col < 9; col++) {
                int zombieHealth = 0;
                for (Zombie zombie : gameState.getZombiesInLane(lane)) {
                    if (zombie.getColumn() == col) {
                        zombieHealth += zombie.getHealth();
                    }
                }
                state[index++] = zombieHealth;
            }
        }
        
        // Game info
        state[index++] = gameState.getSunScore();
        state[index++] = gameState.getSuns().size();
        
        // Card cooldowns
        for (int i = 0; i < 4; i++) {
            state[index++] = gameState.getCardCooldown(i);
        }
        
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
}
