package ui;

import entities.plants.Walnut;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Color;
import java.awt.Font;
import java.awt.Rectangle;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JLayeredPane;
import core.Game;
import entities.plants.FreezePeashooter;
import entities.plants.Peashooter;
import entities.plants.Plant;
import entities.plants.Sunflower;
import utils.AssetManager;

public class GameWindow {
    // 常量定义
    private static final int SUN_LABEL_X = 37;
    private static final int SUN_LABEL_Y = 80;
    private static final int SUN_LABEL_WIDTH = 100;
    private static final int SUN_LABEL_HEIGHT = 20;

    private static final int TRAINING_TOGGLE_X = 955;
    private static final int TRAINING_TOGGLE_Y = 0;
    private static final int TRAINING_TOGGLE_WIDTH = 30;
    private static final int TRAINING_TOGGLE_HEIGHT = 30;

    private static final int TRAINING_LABEL_X = 855;
    private static final int TRAINING_LABEL_Y = 0;
    private static final int TRAINING_LABEL_WIDTH = 100;
    private static final int TRAINING_LABEL_HEIGHT = 30;

    private JFrame frame;
    private Game game;
    private JLabel sunScoreLabel;
    private JLayeredPane layeredPane;
    private PlantCard[] plantCards;
    private Class<? extends Plant> selectedPlant;
    private int selectedPlantCardIndex = -1;

    // 训练模式相关
    private Rectangle trainingToggleRect;
    private boolean trainingMode = false;
    private JLabel trainingLabel;

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
        sunScoreLabel.setBounds(SUN_LABEL_X, SUN_LABEL_Y, SUN_LABEL_WIDTH, SUN_LABEL_HEIGHT);
        sunScoreLabel.setFont(new Font("Arial", Font.BOLD, 14));
        layeredPane.add(sunScoreLabel, JLayeredPane.POPUP_LAYER);

        initTrainingToggle();
        createPlantCards();

        frame.add(layeredPane);
        frame.pack();
        frame.setVisible(true);

        game.start();
    }

    private void initTrainingToggle() {
        trainingToggleRect = new Rectangle(TRAINING_TOGGLE_X, TRAINING_TOGGLE_Y, TRAINING_TOGGLE_WIDTH, TRAINING_TOGGLE_HEIGHT);

        trainingLabel = new JLabel("Training Mode");
        trainingLabel.setBounds(TRAINING_LABEL_X, TRAINING_LABEL_Y, TRAINING_LABEL_WIDTH, TRAINING_LABEL_HEIGHT);
        trainingLabel.setFont(new Font("Arial", Font.BOLD, 12));
        trainingLabel.setForeground(Color.BLACK);
        layeredPane.add(trainingLabel, JLayeredPane.POPUP_LAYER);

        game.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                if (trainingToggleRect.contains(e.getPoint())) {
                    toggleTrainingMode();
                }
            }
        });
    }

    private void toggleTrainingMode() {
        trainingMode = !trainingMode;
        game.setTrainingMode(trainingMode);
        System.out.println("Training mode: " + (trainingMode ? "ON" : "OFF"));
    }

    private void createPlantCards() {
        plantCards = new PlantCard[4];

        plantCards[0] = new PlantCard(game.getGameState(), 0, AssetManager.getImage("card_sunflower"), "Sunflower", 50, 5);
        plantCards[0].setBounds(110, 8, 64, 90);
        plantCards[0].addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                selectPlant(Sunflower.class, 0);
            }
        });

        plantCards[1] = new PlantCard(game.getGameState(), 1, AssetManager.getImage("card_peashooter"), "Peashooter", 100, 10);
        plantCards[1].setBounds(180, 8, 64, 90);
        plantCards[1].addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                selectPlant(Peashooter.class, 1);
            }
        });

        plantCards[2] = new PlantCard(game.getGameState(), 2, AssetManager.getImage("card_freezepeashooter"), "Freeze Peashooter", 175, 10);
        plantCards[2].setBounds(250, 8, 64, 90);
        plantCards[2].addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                selectPlant(FreezePeashooter.class, 2);
            }
        });

        plantCards[3] = new PlantCard(game.getGameState(), 3, AssetManager.getImage("card_walnut"), "Walnut", 50, 15);
        plantCards[3].setBounds(320, 8, 64, 90);
        plantCards[3].addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                selectPlant(Walnut.class, 3);
            }
        });

        for (PlantCard card : plantCards) {
            layeredPane.add(card, JLayeredPane.POPUP_LAYER);
        }
    }

    public void selectPlant(Class<? extends Plant> plantClass, int cardIndex) {
        clearSelection();
        plantCards[cardIndex].setSelected(true);
        selectedPlantCardIndex = cardIndex;
        System.out.println("Selected Plant: " + plantCards[cardIndex].getPlantName());
        selectedPlant = plantClass;
    }

    public void renderUI(Graphics g) {
        // 更新太阳分数显示
        sunScoreLabel.setText("Sun: " + game.getGameState().getSunScore());

        // 绘制训练模式按钮
        if (trainingMode) {
            g.setColor(Color.GREEN);
            g.fillRect(trainingToggleRect.x, trainingToggleRect.y, trainingToggleRect.width, trainingToggleRect.height);
            g.setColor(Color.BLACK);
            g.drawRect(trainingToggleRect.x, trainingToggleRect.y, trainingToggleRect.width, trainingToggleRect.height);
            g.drawLine(trainingToggleRect.x, trainingToggleRect.y, trainingToggleRect.x + trainingToggleRect.width, trainingToggleRect.y + trainingToggleRect.height);
            g.drawLine(trainingToggleRect.x + trainingToggleRect.width, trainingToggleRect.y, trainingToggleRect.x, trainingToggleRect.y + trainingToggleRect.height);
        } else {
            g.setColor(Color.WHITE);
            g.fillRect(trainingToggleRect.x, trainingToggleRect.y, trainingToggleRect.width, trainingToggleRect.height);
            g.setColor(Color.BLACK);
            g.drawRect(trainingToggleRect.x, trainingToggleRect.y, trainingToggleRect.width, trainingToggleRect.height);
        }
    }

    public Class<? extends Plant> getSelectedPlant() {
        return selectedPlant;
    }

    public int getSelectedPlantCardIndex() {
        return selectedPlantCardIndex;
    }

    public int getSelectedPlantCardCooldown() {
        if (selectedPlantCardIndex >= 0 && selectedPlantCardIndex < plantCards.length) {
            return plantCards[selectedPlantCardIndex].getCurrentCooldown();
        }
        return 0;
    }

    public void clearSelection() {
        selectedPlant = null;
        for (PlantCard card : plantCards) {
            card.setSelected(false);
        }
        selectedPlantCardIndex = -1;
    }
}
