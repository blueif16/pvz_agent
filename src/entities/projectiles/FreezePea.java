package entities.projectiles;

import core.GameState;
import entities.zombies.Zombie;

/**
 * Freeze pea projectile that slows zombies
 */
public class FreezePea extends Projectile {
    private static final int DAMAGE = 300;
    private static final float SPEED = 300.0f;
    private static final float SLOW_DURATION = 5.0f;
    
    public FreezePea(GameState gameState, float x, float y, int lane) {
        super(gameState, x, y, lane, DAMAGE, SPEED, "freezepea");
    }
    
    @Override
    protected void onHit(Zombie zombie) {
        zombie.takeDamage(damage);
        zombie.slow(SLOW_DURATION);
    }
} 