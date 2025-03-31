package entities.plants;

import java.awt.Graphics;
import java.awt.Image;

import core.GameObject;
import core.GameState;
import utils.AssetManager;


public abstract class Plant extends GameObject {
    protected int health;
    protected int cost;
    protected int lane;
    protected int gridX;
    protected String imageName;
    protected String name;
    protected float cooldown;
    protected float cooldownTimer;
    
    public Plant(GameState gameState, int gridX, int lane, int cost, int health, String imageName, String name) {
        super(gameState, 60 + gridX * 100, 129 + lane * 120, 80, 80);
        this.gridX = gridX;
        this.lane = lane;
        this.cost = cost;
        this.health = health;
        this.imageName = imageName;
        this.name = name;
    }
    
    @Override
    public void update(float deltaTime) {
        if (health <= 0) {
            setActive(false);
            return;
        }
        
        if (cooldownTimer > 0) {
            cooldownTimer -= deltaTime;
        } else {
            cooldownTimer = cooldown;
            action();
        }
    }
    
    @Override
    public void render(Graphics g) {
        Image image = AssetManager.getImage(imageName);
        g.drawImage(image, (int)x, (int)y, null);
    }
    
    public abstract void action();
    
    public void takeDamage(int damage) {
        health -= damage;
    }
    
    public int getLane() {
        return lane;
    }
    
    public int getGridX() {
        return gridX;
    }
    
    public int getCost() {
        return cost;
    }
    
    public int getHealth() {
        return health;
    }

    public String getName() {
        return name;
    }
} 