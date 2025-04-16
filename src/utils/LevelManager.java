package utils;

import java.util.Random;
import core.GameState;
import entities.Sun;
import entities.zombies.ConeHeadZombie;
import entities.zombies.NormalZombie;
import entities.zombies.Zombie;

public class LevelManager {
    private GameState gameState;
    private Random random;
    
    private int currentLevel;
    private int zombiesSpawned;
    private int zombiesRequired;
    
    // Replace Timer with accumulated time tracking
    private float zombieSpawnAccumulator = 0;
    private float sunSpawnAccumulator = 0;
    
    // Spawn intervals in seconds
    private float zombieSpawnInterval;
    private final float SUN_SPAWN_INTERVAL = 20.0f;
    private final float INITIAL_DELAY_ZOMBIE = 7.0f;
    private final float INITIAL_DELAY_SUN = 5.0f;
    private boolean initialZombieDelayPassed = false;
    private boolean initialSunDelayPassed = false;
    
    public LevelManager(GameState gameState) {
        this.gameState = gameState;
        this.random = new Random();
    }
    
    public void loadLevel(int level) {
        currentLevel = level;
        zombiesSpawned = 0;
        
        switch (level) {
            case 1:
                zombiesRequired = 10;
                break;
            case 2:
                zombiesRequired = 20;
                break;
            default:
                zombiesRequired = 10 + (level - 1) * 10;
                break;
        }
        
        // Reset accumulators and delays
        zombieSpawnAccumulator = 0;
        sunSpawnAccumulator = 0;
        initialZombieDelayPassed = false;
        initialSunDelayPassed = false;
        
        // Calculate zombie spawn interval based on level
        zombieSpawnInterval = Math.max(2.0f, 20.0f - (currentLevel * 1.0f));
    }
    
    private void spawnSun() {
        int x = 100 + random.nextInt(800);
        Sun sun = new Sun(gameState, x, 0, 100 + random.nextInt(400));
        gameState.addGameObject(sun);
    }
    
    private void spawnZombie() {
        if (zombiesSpawned < zombiesRequired) {
            int lane = random.nextInt(5);
            
            Zombie zombie;
            if (currentLevel >= 2 && random.nextInt(100) < 20 + (currentLevel * 5)) {
                zombie = new ConeHeadZombie(gameState, lane);
            } else {
                zombie = new NormalZombie(gameState, lane);
            }
            
            gameState.addGameObject(zombie);
            zombiesSpawned++;
        }
    }
    
    public void update(float deltaTime) {
        // Handle sun spawning
        if (!initialSunDelayPassed) {
            sunSpawnAccumulator += deltaTime;
            if (sunSpawnAccumulator >= INITIAL_DELAY_SUN) {
                initialSunDelayPassed = true;
                sunSpawnAccumulator = 0;
                spawnSun(); // Spawn first sun after initial delay
            }
        } else {
            sunSpawnAccumulator += deltaTime;
            if (sunSpawnAccumulator >= SUN_SPAWN_INTERVAL) {
                sunSpawnAccumulator -= SUN_SPAWN_INTERVAL; // Subtract instead of reset to maintain precision
                spawnSun();
            }
        }
        
        // Handle zombie spawning
        if (!initialZombieDelayPassed) {
            zombieSpawnAccumulator += deltaTime;
            if (zombieSpawnAccumulator >= INITIAL_DELAY_ZOMBIE) {
                initialZombieDelayPassed = true;
                zombieSpawnAccumulator = 0;
                spawnZombie(); // Spawn first zombie after initial delay
            }
        } else if (zombiesSpawned < zombiesRequired) {
            zombieSpawnAccumulator += deltaTime;
            if (zombieSpawnAccumulator >= zombieSpawnInterval) {
                zombieSpawnAccumulator -= zombieSpawnInterval; // Subtract instead of reset to maintain precision
                spawnZombie();
            }
        }
        
        // Check for level completion
        if (gameState.getProgress() >= zombiesRequired * 10) {
            loadLevel(currentLevel + 1);
        }
    }
    
    public int getCurrentLevel() {
        return currentLevel;
    }
    
    public int getZombiesSpawned() {
        return zombiesSpawned;
    }
    
    public int getZombiesRequired() {
        return zombiesRequired;
    }
} 