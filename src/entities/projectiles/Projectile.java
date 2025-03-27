package entities.projectiles;

import java.awt.Graphics;
import java.awt.Image;
import java.util.List;

import core.GameObject;
import core.GameState;
import entities.zombies.Zombie;
import utils.AssetManager;

/**
 * Base class for all projectiles
 */
public abstract class Projectile extends GameObject {
    protected int damage;
    protected float speed;
    protected int lane;
    protected String imageName;
    
    public Projectile(GameState gameState, float x, float y, int lane, int damage, float speed, String imageName) {
        super(gameState, x, y, 28, 28);
        this.lane = lane;
        this.damage = damage;
        this.speed = speed;
        this.imageName = imageName;
    }
    
    @Override
    public void update(float deltaTime) {
        // Move forward
        x += speed * deltaTime;
        
        // Check for zombie collisions
        Zombie hitZombie = checkZombieCollision();
        
        if (hitZombie != null) {
            onHit(hitZombie);
            setActive(false);
        }
        
        // Remove if off screen
        if (x > Game.WIDTH) {
            setActive(false);
        }
    }
    
    @Override
    public void render(Graphics g) {
        Image image = AssetManager.getImage(imageName);
        g.drawImage(image, (int)x, (int)y, null);
    }
    
    protected Zombie checkZombieCollision() {
        List<Zombie> zombiesInLane = gameState.getZombiesInLane(lane);
        
        for (Zombie zombie : zombiesInLane) {
            if (isColliding(zombie)) {
                return zombie;
            }
        }
        
        return null;
    }
    
    protected abstract void onHit(Zombie zombie);
    
    public int getLane() {
        return lane;
    }
} 