package entities.zombies;

import java.awt.Graphics;
import java.awt.Image;
import java.util.List;

import core.GameObject;
import core.GameState;
import entities.plants.Plant;
import utils.AssetManager;

/**
 * Base class for all zombies
 */
public abstract class Zombie extends GameObject {
    protected int health;
    protected float speed;
    protected int damage;
    protected int lane;
    protected String imageName;
    protected boolean slowed;
    protected float slowTimer;
    
    public Zombie(GameState gameState, int lane, int health, float speed, int damage, String imageName) {
        super(gameState, 1000, 109 + lane * 120, 100, 120);
        this.lane = lane;
        this.health = health;
        this.speed = speed;
        this.damage = damage;
        this.imageName = imageName;
        this.slowed = false;
        this.slowTimer = 0;
    }
    
    @Override
    public void update(float deltaTime) {
        if (health <= 0) {
            setActive(false);
            gameState.addProgress(10);
            return;
        }
        
        // Update slow effect
        if (slowed) {
            slowTimer -= deltaTime;
            if (slowTimer <= 0) {
                slowed = false;
            }
        }
        
        // Check for plant collisions
        Plant collidedPlant = checkPlantCollision();
        
        if (collidedPlant != null) {
            // Attack plant
            collidedPlant.takeDamage(damage * deltaTime);
        } else {
            // Move forward
            float actualSpeed = slowed ? speed * 0.5f : speed;
            x -= actualSpeed * deltaTime;
        }
        
        // Check if zombie reached the house
        if (x < 0) {
            // Game over
            System.out.println("ZOMBIES ATE YOUR BRAIN!");
            // TODO: Implement game over logic
        }
    }
    
    @Override
    public void render(Graphics g) {
        Image image = AssetManager.getImage(imageName);
        g.drawImage(image, (int)x, (int)y, null);
    }
    
    private Plant checkPlantCollision() {
        List<Plant> plantsInLane = gameState.getPlantsInLane(lane);
        
        for (Plant plant : plantsInLane) {
            if (isColliding(plant)) {
                return plant;
            }
        }
        
        return null;
    }
    
    public void takeDamage(int damage) {
        health -= damage;
    }
    
    public void slow(float duration) {
        slowed = true;
        slowTimer = duration;
    }
    
    public int getLane() {
        return lane;
    }
    
    public int getHealth() {
        return health;
    }
} 