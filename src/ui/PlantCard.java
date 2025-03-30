package ui;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;

import javax.swing.JPanel;


public class PlantCard extends JPanel {
    private static final long serialVersionUID = 1L;
    
    private Image cardImage;
    private String plantName;
    private int cost;
    private boolean selected;
    
    public PlantCard(Image cardImage, String plantName, int cost) {
        this.cardImage = cardImage;
        this.plantName = plantName;
        this.cost = cost;
        this.selected = false;
        
        setOpaque(false);
    }
    
    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        
        g.drawImage(cardImage, 0, 0, getWidth() , getHeight() , null);
    }
    
    public void setSelected(boolean selected) {
        this.selected = selected;
        repaint();
    }
    
    public boolean isSelected() {
        return selected;
    }
    
    public int getCost() {
        return cost;
    }
} 