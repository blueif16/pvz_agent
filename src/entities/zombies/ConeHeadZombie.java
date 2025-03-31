package entities.zombies;

import core.GameState;


public class ConeHeadZombie extends Zombie {
    private static final int HEALTH = 1800;
    private static final float SPEED = 20.0f;
    private static final int DAMAGE = 130;
    
    public ConeHeadZombie(GameState gameState, int lane) {
        super(gameState, lane, HEALTH, SPEED, DAMAGE, "coneheadzombie");
    }
} 