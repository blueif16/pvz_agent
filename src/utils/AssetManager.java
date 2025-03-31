package utils;

import java.awt.Image;
import java.util.HashMap;
import java.util.Map;

import javax.swing.ImageIcon;


public class AssetManager {
    private static Map<String, Image> images = new HashMap<>();
    
    public static void loadAssets() {
        loadImage("background", "images/mainBG.png");
        
        loadImage("peashooter", "images/plants/peashooter.gif");
        loadImage("freezepeashooter", "images/plants/freezepeashooter.gif");
        loadImage("sunflower", "images/plants/sunflower.gif");
        loadImage("walnut", "images/plants/walnut.gif");

        
        loadImage("card_peashooter", "images/cards/card_peashooter.png");
        loadImage("card_freezepeashooter", "images/cards/card_freezepeashooter.png");
        loadImage("card_sunflower", "images/cards/card_sunflower.png");
        loadImage("card_walnut", "images/cards/card_walnut.png");

        loadImage("pea", "images/pea.png");
        loadImage("freezepea", "images/freezepea.png");
        
        loadImage("normalzombie", "images/zombies/zombie1.gif");
        loadImage("coneheadzombie", "images/zombies/zombie2.gif");
        
        loadImage("sun", "images/sun.png");

    }
    
    private static void loadImage(String key, String path) {
        try {
            Image image = new ImageIcon(AssetManager.class.getResource("/" + path)).getImage();
            images.put(key, image);
        } catch (Exception e) {
            System.err.println("Failed to load image: " + path);
            e.printStackTrace();
        }
    }
    
    public static Image getImage(String key) {
        return images.get(key);
    }
} 