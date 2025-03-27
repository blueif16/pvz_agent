package utils;

import java.awt.Rectangle;

/**
 * Utility class for collision detection
 */
public class CollisionManager {
    
    /**
     * Checks if two rectangles are colliding
     */
    public static boolean isColliding(Rectangle rect1, Rectangle rect2) {
        return rect1.intersects(rect2);
    }
    
    /**
     * Checks if a point is inside a rectangle
     */
    public static boolean isPointInside(int x, int y, Rectangle rect) {
        return rect.contains(x, y);
    }
    
    /**
     * Checks if a point is inside a grid cell
     */
    public static boolean isPointInGridCell(int x, int y, int gridX, int gridY) {
        int cellX = 60 + gridX * 100;
        int cellY = 129 + gridY * 120;
        
        return x >= cellX && x < cellX + 100 && y >= cellY && y < cellY + 120;
    }
} 