package entities.plants;

import core.GameState;

public class Walnut extends Plant {
    private static final int COST = 50;
    private static final int HEALTH = 1000;

    public Walnut(GameState gameState, int gridX, int lane) {
      super(gameState, gridX, lane, COST, HEALTH, "walnut", "Walnut");
      this.cooldownTimer = cooldown;
    } 

    @Override
    public void action() {

    }
}
