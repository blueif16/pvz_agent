package entities.plants;

import core.GameState;
import entities.Sun;

/**
 * Sunflower plant that produces sun
 */
public class Sunflower extends Plant {
    private static final int COST = 50;
    private static final int HEALTH = 200;
    private static final float SUN_PRODUCTION_COOLDOWN = 15.0f;
    
    public Sunflower(GameState gameState, int gridX, int lane) {
        super(gameState, gridX, lane, COST, HEALTH, "sunflower");
        this.cooldown = SUN_PRODUCTION_COOLDOWN;
        this.cooldownTimer = cooldown;
    }
    
    @Override
    public void action() {
        // Create a new sun at this position
        Sun sun = new Sun(gameState, x, y, y + 20);
        gameState.addGameObject(sun);
    }
} 