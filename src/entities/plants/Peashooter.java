package entities.plants;

import core.GameState;
import entities.projectiles.Pea;

public class Peashooter extends Plant {
    private static final int COST = 100;
    private static final int HEALTH = 200;
    private static final float SHOOT_COOLDOWN = 2.0f;
    
    public Peashooter(GameState gameState, int gridX, int lane) {
        super(gameState, gridX, lane, COST, HEALTH, "peashooter");
        this.cooldown = SHOOT_COOLDOWN;
        this.cooldownTimer = cooldown;
    }
    
    @Override
    public void action() {
        if (!gameState.getZombiesInLane(lane).isEmpty()) {
            Pea pea = new Pea(gameState, x + width, y, lane);
            gameState.addGameObject(pea);
        }
    }
} 