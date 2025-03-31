package ui;

import entities.plants.Walnut;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import core.Game;
import core.GameState;
import entities.Sun;
import entities.plants.FreezePeashooter;
import entities.plants.Peashooter;
import entities.plants.Plant;
import entities.plants.Sunflower;
import entities.plants.Walnut;


public class InputHandler extends MouseAdapter {
    private Game game;
    private GameState gameState;
    private int gridX, gridY;
    
    public InputHandler(Game game, GameState gameState) {
        this.game = game;
        this.gameState = gameState;
        
        game.addMouseListener(this);
        game.addMouseMotionListener(this);
    }
    
    @Override
    public void mousePressed(MouseEvent e) {
        int x = e.getX();
        int y = e.getY();
        
        for (Sun sun : gameState.getSuns()) {
            if (x >= sun.getX() && x <= sun.getX() + sun.getWidth() &&
                y >= sun.getY() && y <= sun.getY() + sun.getHeight()) {
                sun.collect();
                return;
            }
        }
        
        if (x >= 60 && x <= 960 && y >= 129 && y <= 729) {
            gridX = (x - 60) / 100;
            gridY = (y - 129) / 120;
            
            plantSelected(gridX, gridY);
        }
    }
    
    public void plantSelected(int gridX, int gridY) {
        Class<? extends Plant> selectedPlant = game.getWindow().getSelectedPlant();
        
        if (selectedPlant == null) {
            return;
        }
        
        for (Plant plant : gameState.getPlantsInLane(gridY)) {
            if (plant.getGridX() == gridX) {
                return;
            }
        }
        
        Plant plant = null;
        int cost = 0;
        
        if (selectedPlant == Sunflower.class) {
            plant = new Sunflower(gameState, gridX, gridY);
            cost = 50;
        } else if (selectedPlant == Peashooter.class) {
            plant = new Peashooter(gameState, gridX, gridY);
            cost = 100;
        } else if (selectedPlant == FreezePeashooter.class) {
            plant = new FreezePeashooter(gameState, gridX, gridY);
            cost = 175;
        } else if (selectedPlant == Walnut.class) {
            plant = new Walnut(gameState, gridX, gridY);
            cost = 50;
        }
        
        if (plant != null && gameState.spendSun(cost)) {
            gameState.addGameObject(plant);
            gameState.setCardCooldown(
                game.getWindow().getSelectedPlantCardIndex(), 
                game.getWindow().getSelectedPlantCardCooldown()
            );
            game.getWindow().clearSelection();
        }
    }

} 