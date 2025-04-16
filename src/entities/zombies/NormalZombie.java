package entities.zombies;

import core.GameState;

public class NormalZombie extends Zombie {
    private static final int HEALTH = 1000;
    private static final float SPEED = 30.0f;
    private static final int DAMAGE = 100;
    
    public NormalZombie(GameState gameState, int lane) {
        super(gameState, lane, HEALTH, SPEED, DAMAGE, "normalzombie");
    }
} 