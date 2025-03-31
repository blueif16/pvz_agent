package entities.projectiles;

import core.GameState;
import entities.zombies.Zombie;

public class Pea extends Projectile {
    private static final int DAMAGE = 200;
    private static final float SPEED = 300.0f;
    
    public Pea(GameState gameState, float x, float y, int lane) {
        super(gameState, x, y, lane, DAMAGE, SPEED, "pea");
    }
    
    @Override
    protected void onHit(Zombie zombie) {
        zombie.takeDamage(damage);
    }
} 