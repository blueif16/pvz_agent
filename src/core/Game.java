package core;

import java.awt.Canvas;
import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferStrategy;
import java.util.ArrayList;

import ui.GameWindow;
import ui.InputHandler;
import utils.AssetManager;
import utils.LevelManager;


public class Game extends Canvas implements Runnable {
    private static final long serialVersionUID = 1L;
    
    public static final int WIDTH = 1000;
    public static final int HEIGHT = 752;
    public static final String TITLE = "Plants vs Zombies";
    
    private Thread thread;
    private boolean running = false;
    
    private GameState gameState;
    private GameWindow window;
    private InputHandler inputHandler;
    private LevelManager levelManager;

    private RewardCounter rewardCounter;

    private boolean isTraining = false;
    
    public Game() {
        AssetManager.loadAssets();
        
        gameState = new GameState();
        inputHandler = new InputHandler(this, gameState);
        levelManager = new LevelManager(gameState);
        
        window = new GameWindow(WIDTH, HEIGHT, TITLE, this);
        
        levelManager.loadLevel(1);
        rewardCounter = new RewardCounter(gameState);
    }
    
    public synchronized void start() {
        if (running) return;
        
        running = true;
        thread = new Thread(this);
        thread.start();
    }
    
    public synchronized void stop() {
        if (!running) return;
        
        running = false;
        try {
            thread.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
    
    @Override
    public void run() {
        long lastTime = System.nanoTime();
        double amountOfTicks = 60.0;
        double ns = 1000000000 / amountOfTicks;
        double delta = 0;
        long timer = System.currentTimeMillis();
        int frames = 0;

        int cumDelta = 0;
        
        while (running) {
            long now = System.nanoTime();
            delta += (now - lastTime) / ns;
            lastTime = now;

            cumDelta += (now - lastTime) / ns;

            if (cumDelta >= 30 && isTraining) {
                int[] actionMask = rewardCounter.getActionMask();
                int actionIndex = Connector.send(gameState, actionMask, rewardCounter.getReward());
                

                rewardCounter.executeAction(actionIndex, this);
                
                cumDelta = 0;
            }

            while (delta >= 1) {
                update(1.0f / 60.0f);
                delta--;

                rewardCounter.addRewardForSurvive(1);
            }
            
            if (running) {
                render();
                frames++;
            }
            
            if (System.currentTimeMillis() - timer > 1000) {
                timer += 1000;
                gameState.decreaseCooldowns();
//                System.out.println("FPS: " + frames);
                frames = 0;
            }
        }
        
        stop();
    }
    
    private void update(float deltaTime) {
        gameState.update(deltaTime);
        levelManager.update(deltaTime);
    }
    
    private void render() {
        BufferStrategy bs = this.getBufferStrategy();
        if (bs == null) {
            this.createBufferStrategy(3);
            return;
        }
        
        Graphics g = bs.getDrawGraphics();
        
        // Clear screen
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, WIDTH, HEIGHT);
        
        g.drawImage(AssetManager.getImage("background"), 0, 0, null);
        
        // Render game state
        gameState.render(g);
        
        // Render UI
        window.renderUI(g);
        
        g.dispose();
        bs.show();
    }
    
    public static void main(String[] args) {
        new Game().start();
    }
    
    public GameState getGameState() {
        return gameState;
    }

    public GameWindow getWindow() { return window; }
    
    public InputHandler getInputHandler() {
        return inputHandler;
    }
    
    public LevelManager getLevelManager() {
        return levelManager;
    }

    public void setTrainingMode(boolean isTraining) {
        this.isTraining = isTraining;
        if (isTraining) {
            Connector.initConnection();
        } else {
            Connector.closeConnection();
        }
    }

    public boolean isTrainingMode() {
        return isTraining;
    }
} 