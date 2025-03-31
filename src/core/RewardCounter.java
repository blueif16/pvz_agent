package core;

import java.util.ArrayList;
import entities.plants.Plant;

public class RewardCounter {
    private int reward;
    private GameState gameState;
    
    // Total number of actions: 4 plant types * 5 lanes * 9 columns + 1 (collect sun) + 1 (do nothing)
    public static final int ACTION_SPACE_SIZE = 4 * 5 * 9 + 2;
    
    // 1,2 action manually set, planting action start at index 2
    public static final int ACTION_DO_NOTHING = 0;
    public static final int ACTION_COLLECT_SUN = 1;

    
    public RewardCounter(GameState gs) {
        reward = 0;
        gameState = gs;
    }
    
    public void collectSun() {
        if (!gameState.getSuns().isEmpty()) {
            gameState.removeGameObject(gameState.getSuns().get(0));
        }
        reward += 5;
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
        game.getInputHandler().plantSelected(col, lane);
    }
    
    public void addRewardForSurvive(int timeDelta) {
        reward += timeDelta;
    }
    
    public void setReward(int incre) {
        reward += incre;
    }
    
    public int getReward() {
        return reward;
    }
    
    public GameState getGameState() {
        return gameState;
    }
    
    public void resetReward() {
        reward = 0;
    }
    
    public static int getActionSpaceSize() {
        return ACTION_SPACE_SIZE;
    }
}
