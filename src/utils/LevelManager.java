package utils;

import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import core.GameState;
import entities.Sun;
import entities.zombies.ConeHeadZombie;
import entities.zombies.NormalZombie;
import entities.zombies.Zombie;

public class LevelManager {
    private GameState gameState;
    private Timer zombieSpawnTimer;
    private Timer sunSpawnTimer;
    private Random random;

    private int currentLevel;
    private int zombiesSpawned;
    private int zombiesRequired;

    // 关卡完成和胜利提示相关字段
    private boolean levelComplete = false;
    private String winMessage = "";

    // 新增关卡提示信息字段
    private String levelPromptMessage = "";

    // 关卡计时（此处主要以僵尸进度作为进度指标）
    private long levelStartTime;
    private long levelDuration; // 单位：毫秒

    public LevelManager(GameState gameState) {
        this.gameState = gameState;
        this.random = new Random();

        zombieSpawnTimer = new Timer();
        sunSpawnTimer = new Timer();
    }

    public void loadLevel(int level) {
        currentLevel = level;
        zombiesSpawned = 0;

        // 重置游戏进度
        gameState.resetProgress();

        if (zombieSpawnTimer != null) {
            zombieSpawnTimer.cancel();
        }
        if (sunSpawnTimer != null) {
            sunSpawnTimer.cancel();
        }
        zombieSpawnTimer = new Timer();
        sunSpawnTimer = new Timer();

        zombiesRequired = 10 + (level - 1) * 15;

        // 初始化关卡计时（例如60秒）
        levelStartTime = System.currentTimeMillis();
        levelDuration = 60000;

        // 重置关卡完成标志和胜利提示
        levelComplete = false;
        winMessage = "";

        // 设置关卡提示信息，并在3秒后清除
        levelPromptMessage = (currentLevel == 1) ? "进入第一关" : "进入下一关";
        new Timer().schedule(new TimerTask(){
            @Override
            public void run() {
                levelPromptMessage = "";
            }
        }, 3000);

        startSunProduction();
        startZombieSpawning();
    }

    private void startSunProduction() {
        long delay = 5000;
        long period = 10000 + (currentLevel - 1) * 1000;
        sunSpawnTimer.schedule(new TimerTask() {
            @Override
            public void run() {
                spawnSun();
            }
        }, delay, period);
    }

    private void startZombieSpawning() {
        // 第一关初始延迟10秒，其它关卡延迟5秒
        int initialDelay = (currentLevel == 1) ? 10000 : 5000;
        zombieSpawnTimer.schedule(new TimerTask() {
            @Override
            public void run() {
                if (zombiesSpawned < zombiesRequired) {
                    spawnZombie();
                    zombiesSpawned++;
                } else {
                    zombieSpawnTimer.cancel();
                }
            }
        }, initialDelay, getZombieSpawnInterval());
    }

    // 调整生成间隔公式：较短的间隔，使僵尸更频繁出现
    private long getZombieSpawnInterval() {
        return Math.max(1000, 10000 - (currentLevel * 500));
    }

    private void spawnSun() {
        int x = 100 + random.nextInt(800);
        Sun sun = new Sun(gameState, x, 0, 100 + random.nextInt(400));
        gameState.addGameObject(sun);
    }

    private void spawnZombie() {
        int lane = random.nextInt(5);

        Zombie zombie;
        int coneHeadChance = Math.min(70, 20 + currentLevel * 7);
        if (currentLevel >= 2 && random.nextInt(100) < coneHeadChance) {
            zombie = new ConeHeadZombie(gameState, lane);
        } else {
            zombie = new NormalZombie(gameState, lane);
        }

        gameState.addGameObject(zombie);
    }

    public void update(float deltaTime) {
        // 检查是否有僵尸越界（假设 zombie.getColumn() < 0 表示僵尸已越过最后一格）
        for (int lane = 0; lane < 5; lane++) {
            for (Zombie zombie : gameState.getZombiesInLane(lane)) {
                if (zombie.getColumn() < 0) {
                    winMessage = "You Failed";
                    new Timer().schedule(new TimerTask() {
                        @Override
                        public void run() {
                            System.exit(0);
                        }
                    }, 3000);
                    return;
                }
            }
        }

        // 当关卡未完成且进度达到要求时触发关卡完成逻辑
        if (!levelComplete && gameState.getProgress() >= zombiesRequired * 10) {
            if (currentLevel == 1) {
                levelComplete = true;
                winMessage = "You win, Next level";
                new Timer().schedule(new TimerTask() {
                    @Override
                    public void run() {
                        levelComplete = false;
                        winMessage = "";
                        loadLevel(currentLevel + 1);
                    }
                }, 3000);
            } else {
                loadLevel(currentLevel + 1);
            }
        }
    }

    public int getCurrentLevel() {
        return currentLevel;
    }

    public int getZombiesSpawned() {
        return zombiesSpawned;
    }

    public int getZombiesRequired() {
        return zombiesRequired;
    }

    // 返回描述僵尸进度的百分比字符串信息，例如 "45%"
    public String getZombieProgressInfo() {
        int total = zombiesRequired * 10;
        int progress = gameState.getProgress();
        int percent = 0;
        if(total > 0) {
            percent = (int)((progress * 100.0) / total);
            if (percent > 100) {
                percent = 100;
            }
        }
        return percent + "%";
    }

    public String getWinMessage() {
        return winMessage;
    }

    // 返回当前关卡提示信息
    public String getLevelPromptMessage() {
        return levelPromptMessage;
    }
}
