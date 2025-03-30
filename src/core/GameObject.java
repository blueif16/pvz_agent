package core;

import java.awt.Graphics;
import java.awt.Rectangle;

public abstract class GameObject {
    protected float x, y;
    protected float velocityX, velocityY;
    protected int width, height;
    protected boolean active = true;
    protected GameState gameState;

    public GameObject(GameState gameState, float x, float y, int width, int height) {
        this.gameState = gameState;
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    public abstract void update(float deltaTime);
    public abstract void render(Graphics g);

    public Rectangle getBounds() {
        return new Rectangle((int)x, (int)y, width, height);
    }

    public boolean isColliding(GameObject other) {
        return getBounds().intersects(other.getBounds());
    }

    public boolean isActive() {
        return active;
    }

    public void setActive(boolean active) {
        this.active = active;
    }

    public float getX() {
        return x;
    }

    public void setX(float x) {
        this.x = x;
    }

    public float getY() {
        return y;
    }

    public void setY(float y) {
        this.y = y;
    }

    public int getWidth() {
        return width;
    }

    public int getHeight() {
        return height;
    }
} 