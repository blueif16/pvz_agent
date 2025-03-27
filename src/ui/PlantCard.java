package ui;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;

import javax.swing.JPanel;

/**
 * UI component for a plant card in the selector
 */
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
        
        // Draw card background
        if (selected) {
            g.setColor(new Color(255, 255, 200));
        } else {
            g.setColor(new Color(220, 220, 220));
        }
        g.fillRect(0, 0, getWidth(), getHeight());
        
        // Draw border
        g.setColor(Color.BLACK);
        g.drawRect(0, 0, getWidth() - 1, getHeight() - 1);
        
        // Draw plant image
        g.drawImage(cardImage, 2, 2, getWidth() - 4, getHeight() - 20, null);
        
        // Draw cost
        g.setColor(Color.BLACK);
        g.setFont(new Font("Arial", Font.BOLD, 12));
        g.drawString(String.valueOf(cost), getWidth() / 2 - 5, getHeight() - 5);
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