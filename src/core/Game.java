package core;

import java.awt.Canvas;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.image.BufferStrategy;
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
        double trainingDelta = 0; // 用于训练模式计时累加

        while (running) {
            long now = System.nanoTime();
            long elapsed = now - lastTime;
            lastTime = now;

            double elapsedTicks = elapsed / ns;
            delta += elapsedTicks;
            trainingDelta += elapsedTicks;

            // 每30个tick触发一次训练操作（仅在训练模式下）
            if (trainingDelta >= 30 && isTraining) {
                int[] actionMask = rewardCounter.getActionMask();
                int actionIndex = Connector.send(gameState, actionMask, rewardCounter.getReward());
                rewardCounter.executeAction(actionIndex, this);
                trainingDelta = 0;
            }

            while (delta >= 1) {
                update(1.0f / 60.0f);
                delta--;
                rewardCounter.addRewardForSurvive(1);
            }

            render();
            frames++;

            if (System.currentTimeMillis() - timer > 1000) {
                timer += 1000;
                gameState.decreaseCooldowns();
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

        // 清屏
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, WIDTH, HEIGHT);

        g.drawImage(AssetManager.getImage("background"), 0, 0, null);

        // 绘制游戏状态
        gameState.render(g);

        // 绘制 GameWindow 内部的 UI（如植物卡片、太阳分数、训练模式按钮等）
        window.renderUI(g);

        // 在右上角显示当前关卡和僵尸进度（放在 training mode 下方）
        String levelInfo = "关卡：" + levelManager.getCurrentLevel();
        String progressInfo = levelManager.getZombieProgressInfo();  // 如 "45%"
        int boxX = 855;
        int boxY = 35;  // 训练模式区域（0~30）下方的位置
        int boxWidth = 150;
        int boxHeight = 40;
        g.setColor(new Color(0, 0, 0, 150));
        g.fillRect(boxX, boxY, boxWidth, boxHeight);
        g.setColor(Color.YELLOW);
        g.setFont(new Font("Arial", Font.BOLD, 14));
        g.drawString(levelInfo, boxX + 5, boxY + 15);
        g.drawString(progressInfo, boxX + 5, boxY + 35);

        // 显示关卡提示信息（例如“进入下一关”）
        String prompt = levelManager.getLevelPromptMessage();
        if (!prompt.isEmpty()) {
            g.setColor(Color.WHITE);
            g.drawString(prompt, 10, 90);
        }

        // 显示胜利或失败提示（红色加粗）
        String winMsg = levelManager.getWinMessage();
        if (winMsg != null && !winMsg.isEmpty()) {
            Font oldFont = g.getFont();
            Font newFont = new Font("Arial", Font.BOLD, 36);
            g.setFont(newFont);
            g.setColor(Color.RED);
            int strWidth = g.getFontMetrics().stringWidth(winMsg);
            int strHeight = g.getFontMetrics().getHeight();
            int x = (WIDTH - strWidth) / 2;
            int y = (HEIGHT - strHeight) / 2 + strHeight;
            g.drawString(winMsg, x, y);
            g.setFont(oldFont);
        }

        g.dispose();
        bs.show();
    }

    public static void main(String[] args) {
        new Game().start();
    }

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
