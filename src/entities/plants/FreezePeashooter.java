package entities.plants;

import core.GameState;
import entities.projectiles.FreezePea;

/**
 * Freeze Peashooter plant that shoots freeze peas
 */
public class FreezePeashooter extends Plant {
    private static final int COST = 175;
    private static final int HEALTH = 200;
    private static final float SHOOT_COOLDOWN = 2.0f;
    
    public FreezePeashooter(GameState gameState, int gridX, int lane) {
        super(gameState, gridX, lane, COST, HEALTH, "freezepeashooter");
        this.cooldown = SHOOT_COOLDOWN;
        this.cooldownTimer = cooldown;
    }
    
    @Override
    public void action() {
        // Only shoot if there are zombies in this lane
        if (!gameState.getZombiesInLane(lane).isEmpty()) {
            FreezePea freezePea = new FreezePea(gameState, x + width, y + height/2, lane);
            gameState.addGameObject(freezePea);
        }
    }
} 