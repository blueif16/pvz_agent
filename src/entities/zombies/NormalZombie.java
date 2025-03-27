package entities.zombies;

import core.GameState;

/**
 * Normal zombie with standard health and speed
 */
public class NormalZombie extends Zombie {
    private static final int HEALTH = 1000;
    private static final float SPEED = 20.0f;
    private static final int DAMAGE = 10;
    
    public NormalZombie(GameState gameState, int lane) {
        super(gameState, lane, HEALTH, SPEED, DAMAGE, "normalzombie");
    }
} 