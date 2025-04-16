package ui;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;

import core.GameState;

import javax.swing.JPanel;


public class PlantCard extends JPanel {
    private static final long serialVersionUID = 1L;
    
    private Image cardImage;
    private String plantName;
    private int cost;
    private boolean selected;
    private final int cooldown;
    private GameState gameState;
    private int index;
    
    public PlantCard(GameState gameState, int index, Image cardImage, String plantName, int cost, int cooldown) {
        this.gameState = gameState;
        this.index = index;
        this.cardImage = cardImage;
        this.plantName = plantName;
        this.cost = cost;
        this.cooldown = cooldown;
        this.selected = false;
        
        setOpaque(false);
    }
    
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        
        g.drawImage(cardImage, 0, 0, getWidth() , getHeight() , null);
    }
    
    public boolean setSelected(boolean selected) {

        // if (cooldown > 0){
        //     return false;
        // }
        if (gameState.getCardCooldown(index) > 0){
            return false;
        }
        this.selected = selected;
        return true;
    }
    
    public boolean isSelected() {
        return selected;
    }
    
    public int getCost() {
        return cost;
    }

    public int getCooldownSetting() {
        return cooldown;
    }

    public float getCurrentCooldown() {
        return gameState.getCardCooldown(index);
    }

    public String getPlantName() {return plantName;}
} 