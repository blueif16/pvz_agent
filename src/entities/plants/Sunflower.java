package entities.plants;

import core.GameState;
import entities.Sun;

public class Sunflower extends Plant {
    private static final int COST = 50;
    private static final int HEALTH = 200;
    private static final float SUN_PRODUCTION_COOLDOWN = 15.0f;
    
    public Sunflower(GameState gameState, int gridX, int lane) {
        super(gameState, gridX, lane, COST, HEALTH, "sunflower", "Sunflower");
        this.cooldown = SUN_PRODUCTION_COOLDOWN;
        this.cooldownTimer = cooldown;
    }
    
    @Override
    public void action() {
        Sun sun = new Sun(gameState, x, y, y + 20);
        gameState.addGameObject(sun);
    }
} 