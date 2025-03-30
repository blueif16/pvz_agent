package utils;

import java.awt.Rectangle;


public class CollisionManager {
    
    
    public static boolean isColliding(Rectangle rect1, Rectangle rect2) {
        return rect1.intersects(rect2);
    }

    public static boolean isPointInside(int x, int y, Rectangle rect) {
        return rect.contains(x, y);
    }
    
    public static boolean isPointInGridCell(int x, int y, int gridX, int gridY) {
        int cellX = 60 + gridX * 100;
        int cellY = 129 + gridY * 120;
        
        return x >= cellX && x < cellX + 100 && y >= cellY && y < cellY + 120;
    }
} 