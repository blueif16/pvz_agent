package core;

import java.awt.Graphics;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import entities.plants.Plant;
import entities.projectiles.Projectile;
import entities.zombies.Zombie;
import entities.Sun;
import utils.LevelManager;

@SuppressWarnings("unchecked")

public class GameState {
    private List<GameObject> gameObjects;
    private List<Plant>[] plants;
    private List<Zombie>[] zombies;
    private List<Projectile>[] projectiles;
    private List<Sun> suns;
    private float[] cooldowns;

    private int sunScore = 150;
    private int progress = 0;
    private int currentLevel = 1;

    private double reward = 0;

    private boolean terminated = false;
    private LevelManager levelManager;
    private Game game;
    private String terminationMessage = "";
    

    public GameState() {
        gameObjects = new CopyOnWriteArrayList<>();
        
        plants = new CopyOnWriteArrayList[5];
        zombies = new CopyOnWriteArrayList[5];
        projectiles = new CopyOnWriteArrayList[5];
        
        for (int i = 0; i < 5; i++) {
            plants[i] = new CopyOnWriteArrayList<>();
            zombies[i] = new CopyOnWriteArrayList<>();
            projectiles[i] = new CopyOnWriteArrayList<>();
        }
        
        suns = new CopyOnWriteArrayList<>();
        cooldowns = new float[4];

        for (int i = 0; i < 4; i++) {
            cooldowns[i] = 0;
        }
    }
    
    // Method to set the Game reference
    public void setGame(Game game) {
        this.game = game;
    }
    
    // Method to get the Game reference
    public Game getGame() {
        return game;
    }
    
    // Method to set the LevelManager reference
    public void setLevelManager(LevelManager levelManager) {
        this.levelManager = levelManager;
    }
    
    // Method to get the LevelManager reference
    public LevelManager getLevelManager() {
        return levelManager;
    }

    public void decreaseCooldowns(float deltaTime) {
        for (int i = 0; i < 4; i++) {
            if (cooldowns[i] > 0) {
                // System.out.println("Cooldown decreased for " + i + " by " + deltaTime);
                cooldowns[i] = Math.max(0, cooldowns[i] - deltaTime);
            }
        }
    }
    
    public void update(float deltaTime) {
        // Update all game objects
        for (GameObject obj : gameObjects) {
            if (obj.isActive()) {
                obj.update(deltaTime);
            }
        }
        
        // Create temporary lists for items to remove
        List<GameObject> objectsToRemove = new ArrayList<>();
        List<Plant>[] plantsToRemove = new ArrayList[5];
        List<Zombie>[] zombiesToRemove = new ArrayList[5];
        List<Projectile>[] projectilesToRemove = new ArrayList[5];
        List<Sun> sunsToRemove = new ArrayList<>();
        
        // Initialize removal lists
        for (int i = 0; i < 5; i++) {
            plantsToRemove[i] = new ArrayList<>();
            zombiesToRemove[i] = new ArrayList<>();
            projectilesToRemove[i] = new ArrayList<>();
        }
        
        // Find inactive objects
        for (GameObject obj : gameObjects) {
            if (!obj.isActive()) {
                objectsToRemove.add(obj);
            }
        }
        
        // Find inactive lane-specific objects
        for (int i = 0; i < 5; i++) {
            for (Plant plant : plants[i]) {
                if (!plant.isActive()) {
                    plantsToRemove[i].add(plant);
                }
            }
            
            for (Zombie zombie : zombies[i]) {
                if (!zombie.isActive()) {
                    zombiesToRemove[i].add(zombie);
                }
            }
            
            for (Projectile projectile : projectiles[i]) {
                if (!projectile.isActive()) {
                    projectilesToRemove[i].add(projectile);
                }
            }
        }
        
        // Find inactive suns
        for (Sun sun : suns) {
            if (!sun.isActive()) {
                sunsToRemove.add(sun);
            }
        }
        
        // Now remove all inactive objects safely
        gameObjects.removeAll(objectsToRemove);
        
        for (int i = 0; i < 5; i++) {
            plants[i].removeAll(plantsToRemove[i]);
            zombies[i].removeAll(zombiesToRemove[i]);
            projectiles[i].removeAll(projectilesToRemove[i]);
        }
        
        suns.removeAll(sunsToRemove);
        
        decreaseCooldowns(deltaTime);
        
        // If levelManager is set, update it
        if (levelManager != null) {
            levelManager.update(deltaTime);
        }
    }
    
    public void render(Graphics g) {
        for (GameObject obj : gameObjects) {
            if (obj.isActive()) {
                obj.render(g);
            }
        }
    }

    public void terminate() {
        terminated = true;
    }
    
    public boolean isTerminated() {
        return terminated;
    }
    
    public void setTerminationMessage(String message) {
        this.terminationMessage = message;
    }
    
    public String getTerminationMessage() {
        return terminationMessage;
    }
    
    public void addGameObject(GameObject obj) {
        gameObjects.add(obj);
        
        if (obj instanceof Plant) {
            Plant plant = (Plant) obj;
            plants[plant.getLane()].add(plant);
        } else if (obj instanceof Zombie) {
            Zombie zombie = (Zombie) obj;
            zombies[zombie.getLane()].add(zombie);
        } else if (obj instanceof Projectile) {
            Projectile projectile = (Projectile) obj;
            projectiles[projectile.getLane()].add(projectile);
        } else if (obj instanceof Sun) {
            suns.add((Sun) obj);
        }
    }
    
    public void removeGameObject(GameObject obj) {
        gameObjects.remove(obj);
        
        if (obj instanceof Plant) {
            Plant plant = (Plant) obj;
            plants[plant.getLane()].remove(plant);
        } else if (obj instanceof Zombie) {
            Zombie zombie = (Zombie) obj;
            zombies[zombie.getLane()].remove(zombie);
        } else if (obj instanceof Projectile) {
            Projectile projectile = (Projectile) obj;
            projectiles[projectile.getLane()].remove(projectile);
        } else if (obj instanceof Sun) {
            suns.remove(obj);
        }
    }

    // public void reset() {
    //     // Clear all collections
    //     gameObjects.clear();
    //     suns.clear();
        
    //     for (int i = 0; i < 5; i++) {
    //         plants[i].clear();
    //         zombies[i].clear();
    //         projectiles[i].clear();
    //     }
        
    //     // Nullify references
    //     levelManager = null;
    //     game = null;
        
    //     // Reset primitive values
    //     sunScore = 150;
    //     progress = 0;
    //     currentLevel = 1;
    //     reward = 0;
    //     terminated = false;
        
    //     // Help GC
    //     System.gc();
    // }

    public List<Plant> getPlantsInLane(int lane) {
        return plants[lane];
    }
    
    public List<Zombie> getZombiesInLane(int lane) {
        return zombies[lane];
    }
    
    public List<Projectile> getProjectilesInLane(int lane) {
        return projectiles[lane];
    }
    
    public List<Sun> getSuns() {
        return suns;
    }
    
    public int getSunScore() {
        return sunScore;
    }
    
    public void addSunScore(int amount) {
        this.sunScore += amount;
    }

    public void addReward(double amount) {
        this.reward += amount;
    }

    public double getReward() {
        return reward;
    }
    
    public boolean spendSun(int amount) {
        if (sunScore >= amount) {
            sunScore -= amount;
            return true;
        }
        return false;
    }
    
    public int getProgress() {
        return progress;
    }
    
    public void addProgress(int amount) {
        this.progress += amount;
    }
    
    public int getCurrentLevel() {
        return currentLevel;
    }
    
    public void setCurrentLevel(int level) {
        this.currentLevel = level;
    }

    public void setCardCooldown(int index, float cooldown) {
        cooldowns[index] = cooldown;
        // System.out.println("Card cooldown set to " + cooldown + " for index " + index);
    }

    public float getCardCooldown(int index) {
        return cooldowns[index];
    }
} 