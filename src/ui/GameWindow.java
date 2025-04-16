package ui;

import entities.plants.Walnut;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Color;
import java.awt.Font;
import java.awt.Image;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.Rectangle;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JLayeredPane;

import core.Game;
import entities.plants.FreezePeashooter;
import entities.plants.Peashooter;
import entities.plants.Plant;
import entities.plants.Sunflower;
import entities.plants.Walnut;
import utils.AssetManager;


public class GameWindow {
    private JFrame frame;
    private Game game;
    private JLabel sunScoreLabel;
    private JLayeredPane layeredPane;
    private PlantCard[] plantCards;
    private Class<? extends Plant> selectedPlant;
    private int selectedPlantCardIndex;
    
    // Agent mode toggle
    private Rectangle agentToggleRect;
    private boolean agentMode = false;
    private JLabel agentLabel;
    

    public GameWindow(int width, int height, String title, Game game) {
        this.game = game;
        
        frame = new JFrame(title);
        frame.setPreferredSize(new Dimension(width, height));
        frame.setMaximumSize(new Dimension(width, height));
        frame.setMinimumSize(new Dimension(width, height));
        
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        
        layeredPane = new JLayeredPane();
        layeredPane.setPreferredSize(new Dimension(width, height));
        
        game.setBounds(0, 0, width, height);
        layeredPane.add(game, JLayeredPane.DEFAULT_LAYER);
        
        sunScoreLabel = new JLabel("Sun: 150");
        sunScoreLabel.setBounds(37, 80, 100, 20);
        sunScoreLabel.setFont(new Font("Arial", Font.BOLD, 14));
        layeredPane.add(sunScoreLabel, JLayeredPane.POPUP_LAYER);
        
        initToggleButtons();
        
        createPlantCards();
        
        frame.add(layeredPane);
        
        frame.pack();
        frame.setVisible(true);
        
        game.start();
    }
    
    private void initToggleButtons() {
        // Agent mode toggle
        agentToggleRect = new Rectangle(955, 0, 30, 30);
        
        agentLabel = new JLabel("Agent Mode");
        agentLabel.setBounds(855, 0, 100, 30);
        agentLabel.setFont(new Font("Arial", Font.BOLD, 12));
        agentLabel.setForeground(Color.BLACK);
        layeredPane.add(agentLabel, JLayeredPane.POPUP_LAYER);
        
        
        // game.addMouseListener(new MouseAdapter() {
        //     @Override
        //     public void mouseClicked(MouseEvent e) {
        //         if (agentToggleRect.contains(e.getPoint())) {
        //             toggleAgentMode();
        //         } 
        //     }
        // });
    }
    

    
    private void createPlantCards() {
        plantCards = new PlantCard[4];
        
        plantCards[0] = new PlantCard(game.getGameState(), 0, AssetManager.getImage("card_sunflower"), "Sunflower", 50, 5);
        plantCards[0].setBounds(110, 8, 64, 90);
        plantCards[0].addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                selectPlant(Sunflower.class, 0);
            }
        });
        
        plantCards[1] = new PlantCard(game.getGameState(), 1, AssetManager.getImage("card_peashooter"), "Peashooter", 100, 10);
        plantCards[1].setBounds(180, 8, 64, 90);
        plantCards[1].addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                selectPlant(Peashooter.class, 1);
            }
        });
        
        plantCards[2] = new PlantCard(game.getGameState(), 2, AssetManager.getImage("card_freezepeashooter"), "Freeze Peashooter", 175, 10);
        plantCards[2].setBounds(250, 8, 64, 90);
        plantCards[2].addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                selectPlant(FreezePeashooter.class, 2);
            }
        });

        plantCards[3] = new PlantCard(game.getGameState(), 3, AssetManager.getImage("card_walnut"), "Walnut", 50, 15);
        plantCards[3].setBounds(320, 8, 64, 90);
        plantCards[3].addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                selectPlant(Walnut.class, 3);
            }
        });
        
        for (PlantCard card : plantCards) {
            layeredPane.add(card, JLayeredPane.POPUP_LAYER);
        }
    }
    
    public void selectPlant(Class<? extends Plant> plantClass, int cardIndex) {
        boolean success = plantCards[cardIndex].setSelected(true);
        if (success) {
            selectedPlantCardIndex = cardIndex;
            // System.out.println(plantCards[cardIndex].getPlantName() + " " + game.getGameState().getCardCooldown(cardIndex));
            selectedPlant = plantClass;
        }
    }
    
    public void renderUI(Graphics g) {
        sunScoreLabel.setText("Sun: " + game.getGameState().getSunScore());
        
        // Render agent mode toggle
        if (agentMode) {
            g.setColor(Color.GREEN);
            g.fillRect(agentToggleRect.x, agentToggleRect.y, 
                        agentToggleRect.width, agentToggleRect.height);
            g.setColor(Color.BLACK);
            g.drawRect(agentToggleRect.x, agentToggleRect.y, 
                        agentToggleRect.width, agentToggleRect.height);
            g.drawLine(agentToggleRect.x, agentToggleRect.y, 
                        agentToggleRect.x + agentToggleRect.width, 
                        agentToggleRect.y + agentToggleRect.height);
            g.drawLine(agentToggleRect.x + agentToggleRect.width, agentToggleRect.y, 
                        agentToggleRect.x, agentToggleRect.y + agentToggleRect.height);
        } else {
            g.setColor(Color.WHITE);
            g.fillRect(agentToggleRect.x, agentToggleRect.y, 
                        agentToggleRect.width, agentToggleRect.height);
            g.setColor(Color.BLACK);
            g.drawRect(agentToggleRect.x, agentToggleRect.y, 
                        agentToggleRect.width, agentToggleRect.height);
        }
        
    
    }
    
    public Class<? extends Plant> getSelectedPlant() {
        return selectedPlant;
    }

    public int getSelectedPlantCardIndex() {
        return selectedPlantCardIndex;
    }

    public float getSelectedPlantCardCooldown() {
        return plantCards[selectedPlantCardIndex].getCooldownSetting();
    }
    
    public void clearSelection() {
        selectedPlant = null;
        for (PlantCard card : plantCards) {
            card.setSelected(false);
        }
        selectedPlantCardIndex = -1;
    }

    public void setAgentMode(boolean agentMode) {  
        this.agentMode = agentMode;
        // Update the UI to reflect the current mode
        if (agentLabel != null) {
            agentLabel.setForeground(agentMode ? Color.GREEN : Color.BLACK);
        }
    }

    public void reset() {
        // Reset plant cards
        for (PlantCard card : plantCards) {
            card.setSelected(false);
        }
        
    }

} 