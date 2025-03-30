package ui;

import java.awt.BorderLayout;
import java.awt.Canvas;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Color;
import java.awt.Font;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JLayeredPane;

import core.Game;
import core.GameState;
import entities.plants.FreezePeashooter;
import entities.plants.Peashooter;
import entities.plants.Plant;
import entities.plants.Sunflower;
import utils.AssetManager;


public class GameWindow {
    private JFrame frame;
    private Game game;
    private JLabel sunScoreLabel;
    private JLayeredPane layeredPane;
    private PlantCard[] plantCards;
    private Class<? extends Plant> selectedPlant;
    
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
        sunScoreLabel.setForeground(Color.BLACK);
        sunScoreLabel.setFont(new Font("Arial", Font.BOLD, 14));
        sunScoreLabel.setOpaque(false);
        layeredPane.add(sunScoreLabel, JLayeredPane.POPUP_LAYER);
        
        createPlantCards();
        
        frame.add(layeredPane);
        
        frame.pack();
        frame.setVisible(true);
        
        game.start();
    }
    
    private void createPlantCards() {
        plantCards = new PlantCard[3];
        
        plantCards[0] = new PlantCard(AssetManager.getImage("card_sunflower"), "Sunflower", 50);
        plantCards[0].setBounds(110, 8, 64, 90);
        plantCards[0].addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                selectPlant(Sunflower.class, 0);
            }
        });
        
        plantCards[1] = new PlantCard(AssetManager.getImage("card_peashooter"), "Peashooter", 100);
        plantCards[1].setBounds(180, 8, 64, 90);
        plantCards[1].addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                selectPlant(Peashooter.class, 1);
            }
        });
        
        plantCards[2] = new PlantCard(AssetManager.getImage("card_freezepeashooter"), "Freeze Peashooter", 175);
        plantCards[2].setBounds(250, 8, 64, 90);
        plantCards[2].addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                selectPlant(FreezePeashooter.class, 2);
            }
        });
        
        for (PlantCard card : plantCards) {
            layeredPane.add(card, JLayeredPane.POPUP_LAYER);
        }
    }
    
    private void selectPlant(Class<? extends Plant> plantClass, int cardIndex) {
        for (int i = 0; i < plantCards.length; i++) {
            plantCards[i].setSelected(i == cardIndex);
        }
        
        selectedPlant = plantClass;
    }
    
    public void renderUI(Graphics g) {
        sunScoreLabel.setText("Sun: " + game.getGameState().getSunScore());
    }
    
    public Class<? extends Plant> getSelectedPlant() {
        return selectedPlant;
    }
    
    public void clearSelection() {
        selectedPlant = null;
        for (PlantCard card : plantCards) {
            card.setSelected(false);
        }
    }
} 