package ui;

import java.awt.BorderLayout;
import java.awt.Canvas;
import java.awt.Dimension;
import java.awt.Graphics;

import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import core.Game;
import utils.AssetManager;

/**
 * Main game window and UI components
 */
public class GameWindow {
    private JFrame frame;
    private Game game;
    private JLabel sunScoreLabel;
    private PlantSelector plantSelector;
    
    public GameWindow(int width, int height, String title, Game game) {
        this.game = game;
        
        frame = new JFrame(title);
        frame.setPreferredSize(new Dimension(width, height));
        frame.setMaximumSize(new Dimension(width, height));
        frame.setMinimumSize(new Dimension(width, height));
        
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        
        // Create UI panel
        JPanel uiPanel = new JPanel();
        uiPanel.setLayout(null);
        uiPanel.setOpaque(false);
        
        // Sun score display
        sunScoreLabel = new JLabel("Sun: 150");
        sunScoreLabel.setBounds(37, 80, 100, 20);
        uiPanel.add(sunScoreLabel);
        
        // Plant selector
        plantSelector = new PlantSelector(game.getGameState());
        plantSelector.setBounds(110, 8, 300, 90);
        uiPanel.add(plantSelector);
        
        // Add components to frame
        frame.add(game, BorderLayout.CENTER);
        frame.add(uiPanel, BorderLayout.NORTH);
        
        frame.pack();
        frame.setVisible(true);
        
        game.start();
    }
    
    public void renderUI(Graphics g) {
        // Update sun score display
        sunScoreLabel.setText("Sun: " + game.getGameState().getSunScore());
    }
    
    public PlantSelector getPlantSelector() {
        return plantSelector;
    }
} 