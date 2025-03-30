package utils;

import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

import core.GameState;
import entities.Sun;
import entities.zombies.ConeHeadZombie;
import entities.zombies.NormalZombie;
import entities.zombies.Zombie;


public class LevelManager {
    private GameState gameState;
    private Timer zombieSpawnTimer;
    private Timer sunSpawnTimer;
    private Random random;
    
    private int currentLevel;
    private int zombiesSpawned;
    private int zombiesRequired;
    
    public LevelManager(GameState gameState) {
        this.gameState = gameState;
        this.random = new Random();
        
        zombieSpawnTimer = new Timer();
        sunSpawnTimer = new Timer();
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
        
        startSunProduction();
        
        startZombieSpawning();
    }
    
    private void startSunProduction() {
        sunSpawnTimer.schedule(new TimerTask() {
            @Override
            public void run() {
                spawnSun();
            }
        }, 5000, 10000); 
    }
    
    private void startZombieSpawning() {
        zombieSpawnTimer.schedule(new TimerTask() {
            @Override
            public void run() {
                if (zombiesSpawned < zombiesRequired) {
                    spawnZombie();
                    zombiesSpawned++;
                } else {
                    zombieSpawnTimer.cancel();
                }
            }
        }, 10000, getZombieSpawnInterval()); 
    }
    
    private long getZombieSpawnInterval() {
        return Math.max(2000, 20000 - (currentLevel * 1000));
    }
    
    private void spawnSun() {
        int x = 100 + random.nextInt(800);
        Sun sun = new Sun(gameState, x, 0, 100 + random.nextInt(400));
        gameState.addGameObject(sun);
    }
    
    private void spawnZombie() {
        int lane = random.nextInt(5);
        
        Zombie zombie;
        if (currentLevel >= 2 && random.nextInt(100) < 20 + (currentLevel * 5)) {
            zombie = new ConeHeadZombie(gameState, lane);
        } else {
            zombie = new NormalZombie(gameState, lane);
        }
        
        gameState.addGameObject(zombie);
    }
    
    public void update(float deltaTime) {
        
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