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

    private int totalStep = 0;
    private int maxStep;

    private boolean isDone = false;
    private boolean truncated = false;
    
    private boolean headless = false;
    private boolean agentMode = false;
    
    // Default constructor for normal game mode
    public Game() {
        this(100000, false); // Default max steps
    }
    
    // Constructor with maxStep for training mode
    public Game(int maxStep, boolean headless) {
        AssetManager.loadAssets();
        
        gameState = new GameState();
        inputHandler = new InputHandler(this, gameState);
        levelManager = new LevelManager(gameState);
        
        // Set references in GameState
        gameState.setGame(this);
        gameState.setLevelManager(levelManager);
        
        // Only create window if not in headless mode
        // if (!headless) {
        //     window = new GameWindow(WIDTH, HEIGHT, TITLE, this);
        // }

        window = new GameWindow(WIDTH, HEIGHT, TITLE, this);
        
        levelManager.loadLevel(1);
        rewardCounter = new RewardCounter(gameState);

        this.maxStep = maxStep;
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

    public StateAction reset() {
        // Create new GameState instead of resetting
        gameState = new GameState();
        gameState.setGame(this);

        levelManager = new LevelManager(gameState);
        gameState.setLevelManager(levelManager);


        // Recreate dependent components
        inputHandler = new InputHandler(this, gameState);
        rewardCounter = new RewardCounter(gameState);
        
        // Reset level manager
        levelManager.loadLevel(1);
        
        // Reset window if exists
        if (window != null) {
            window.reset();
        }
        
        // Explicitly help GC
        System.gc();

        totalStep = 0;
        isDone = false;
        truncated = false;

        int[] actionMask = rewardCounter.getActionMask();
        float[] state = RewardCounter.processGameState(gameState);
        
        return new StateAction(state, actionMask);
    }

    // step the game with action, return the new state, action mask, reward, done 
    public StepInfo step(int actionIndex) {
        isDone = truncated = false;

        double initialReward = gameState.getReward();
        
        
        // Execute action
        rewardCounter.executeAction(actionIndex, this);
        
        float[] final_obs = null;

        // Run simulation for fixed number of steps
        for (int i = 0; i < 30; i++) {
            update(1.0f / 60.0f);

            if (i % 10 == 0) {
                render();
            }   

            if (gameState.isTerminated()) {
                isDone = true;
                break;
            } 
        }

        // Check if max steps reached
        truncated = ++totalStep >= maxStep;
        double reward;

        // Handle episode end conditions
        if (isDone) {
            // Game over - zombie reached house or all zombies defeated
            // final_obs = RewardCounter.processGameState(gameState);
            reward = gameState.getReward() - initialReward;
            reset(); 
            
        }
        else if (truncated) {
            // Max steps reached
            final_obs = RewardCounter.processGameState(gameState);
            reward = gameState.getReward() - initialReward;
            reset();
            
        }
        else {
            // Add small reward for surviving
            gameState.addReward(0.05);
            reward = gameState.getReward() - initialReward;
        }

        // Get updated state information
        float[] newState = RewardCounter.processGameState(gameState);
        int[] newActionMask = rewardCounter.getActionMask();    

        return new StepInfo(newState, newActionMask, reward, isDone, truncated, final_obs);
    }   
    
    @Override
    public void run() {
        long lastTime = System.nanoTime();
        double amountOfTicks = 60.0;
        double ns = 1000000000 / amountOfTicks;
        double delta = 0;
        long timer = System.currentTimeMillis();
        int updateCount = 0;
        
        while (running) { 
            long now = System.nanoTime();
            delta += (now - lastTime) / ns;
            lastTime = now;
            
            while (delta >= 1) {
                update(1.0f / 60.0f);
                updateCount++;

                if (gameState.isTerminated()) {
                    running = false;
                    break;
                }
                delta--;
            }
            
            if (running) {
                render();
            }
        }
        
        stop();
    }
    
    private void update(float deltaTime) {
        gameState.update(deltaTime);
        levelManager.update(deltaTime);
    }
    
    private void render() {
        // if (headless) {
        //     return; // Skip rendering in headless mode
        // }
        
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
        if (window != null) {
            window.renderUI(g);
//            System.out.println("Rendering, current reward: " + gameState.getReward());
        }
        
        g.dispose();
        bs.show();
    }
    
    public static void main(String[] args) {
        // Parse command line arguments
        boolean agentMode = false;
        boolean headless = false;
        
        for (String arg : args) {
            if (arg.equals("--agent")) {
                agentMode = true;
            } else if (arg.equals("--headless")) {
                headless = true;
            }
        }
        
        // Start in normal game mode by default
        Game game = new Game();
        
        // Apply mode settings if specified
        if (agentMode) {
            game.setAgentMode(true);
        }
        if (headless) {
            game.setHeadless(true);
        }
        
        game.start();
    }
    
    // Getters
    public GameState getGameState() {
        return gameState;
    }

    public GameWindow getWindow() { 
        return window; 
    }
    
    public InputHandler getInputHandler() {
        return inputHandler;
    }
    
    public LevelManager getLevelManager() {
        return levelManager;
    }

    public RewardCounter getRewardCounter() {
        return rewardCounter;
    }

    public void setHeadless(boolean headless) {
        this.headless = headless;
    }

    public void setAgentMode(boolean agentMode) {
        this.agentMode = agentMode;
        
        // Update window UI if window exists
        if (window != null) {
            window.setAgentMode(agentMode);
        }
    }

    public boolean isHeadless() {
        return headless;
    }
} 