package core;

import java.awt.Graphics;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;

import entities.plants.Plant;
import entities.projectiles.Projectile;
import entities.zombies.Zombie;
import entities.Sun;

public class GameState {
    private List<GameObject> gameObjects;
    private List<Plant>[] plants;
    private List<Zombie>[] zombies;
    private List<Projectile>[] projectiles;
    private List<Sun> suns;
    
    private int sunScore = 150;
    private int progress = 0;
    private int currentLevel = 1;
    

    public GameState() {
        gameObjects = new CopyOnWriteArrayList<>();
        
        plants = new ArrayList[5];
        zombies = new ArrayList[5];
        projectiles = new ArrayList[5];
        
        for (int i = 0; i < 5; i++) {
            plants[i] = new ArrayList<>();
            zombies[i] = new ArrayList<>();
            projectiles[i] = new ArrayList<>();
        }
        
        suns = new ArrayList<>();
    }
    
    public void update(float deltaTime) {
        for (GameObject obj : gameObjects) {
            if (obj.isActive()) {
                obj.update(deltaTime);
            }
        }
        
        gameObjects.removeIf(obj -> !obj.isActive());
        
        for (int i = 0; i < 5; i++) {
            plants[i].removeIf(plant -> !plant.isActive());
            zombies[i].removeIf(zombie -> !zombie.isActive());
            projectiles[i].removeIf(projectile -> !projectile.isActive());
        }
        
        suns.removeIf(sun -> !sun.isActive());
    }
    
    public void render(Graphics g) {
        for (GameObject obj : gameObjects) {
            if (obj.isActive()) {
                obj.render(g);
            }
        }
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
} 