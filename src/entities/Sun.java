package entities;

import java.awt.Graphics;
import java.awt.Image;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;

import core.GameObject;
import core.GameState;
import utils.AssetManager;


public class Sun extends GameObject implements MouseListener {
    private static final int SUN_VALUE = 25;
    private static final float FALL_SPEED = 80.0f;
    private static final float LIFETIME = 10.0f;
    
    private float targetY;
    private float lifeTimer;
    private boolean collected;
    
    public Sun(GameState gameState, float x, float y, float targetY) {
        super(gameState, x, y, 80, 80);
        this.targetY = targetY;
        this.lifeTimer = LIFETIME;
        this.collected = false;
    }
    
    @Override
    public void update(float deltaTime) {
        if (collected) {
            setActive(false);
            return;
        }
        
        if (y < targetY) {
            y += FALL_SPEED * deltaTime;
            if (y > targetY) {
                y = targetY;
            }
        } else {
            lifeTimer -= deltaTime;
            if (lifeTimer <= 0) {
                setActive(false);
            }
        }
    }
    
    @Override
    public void render(Graphics g) {
        Image image = AssetManager.getImage("sun");
        g.drawImage(image, (int)x, (int)y, null);
    }
    
    public void collect() {
        if (!collected) {
            collected = true;
            gameState.addSunScore(SUN_VALUE);
        }
    }
    
    @Override
    public void mouseClicked(MouseEvent e) {}
    
    @Override
    public void mousePressed(MouseEvent e) {}
    
    @Override
    public void mouseReleased(MouseEvent e) {
        int mouseX = e.getX();
        int mouseY = e.getY();
        
        if (mouseX >= x && mouseX <= x + width && mouseY >= y && mouseY <= y + height) {
            collect();
        }
    }
    
    @Override
    public void mouseEntered(MouseEvent e) {}
    
    @Override
    public void mouseExited(MouseEvent e) {}
} 