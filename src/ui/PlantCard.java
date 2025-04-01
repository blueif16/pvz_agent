package ui;

import javax.swing.JComponent;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Dimension;

public class PlantCard extends JComponent {
    private int cardIndex;
    private Image cardImage;
    private String plantName;
    private int cost;
    private int cooldown;
    private int currentCooldown;
    private boolean selected = false;
    private boolean available = true;

    public PlantCard(Object gameState, int cardIndex, Image cardImage, String plantName, int cost, int cooldown) {
        this.cardIndex = cardIndex;
        this.cardImage = cardImage;
        this.plantName = plantName;
        this.cost = cost;
        this.cooldown = cooldown;
        this.currentCooldown = 0;
        // 确保整个组件区域都能接收鼠标事件
        setOpaque(true);
        // 设置组件大小，确保点击区域正确
        setPreferredSize(new Dimension(64, 90));
    }

    public String getPlantName() {
        return plantName;
    }

    public int getCost() {
        return cost;
    }

    public int getCurrentCooldown() {
        return currentCooldown;
    }

    public void setSelected(boolean selected) {
        this.selected = selected;
        repaint();
    }

    public void setCurrentCooldown(int cooldown) {
        this.currentCooldown = cooldown;
        repaint();
    }

    public void setAvailable(boolean available) {
        this.available = available;
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        // 填充背景（可选，用于扩大点击区域视觉效果）
        g.setColor(Color.GRAY);
        g.fillRect(0, 0, getWidth(), getHeight());

        // 绘制卡片图像
        g.drawImage(cardImage, 0, 0, getWidth(), getHeight(), null);

        // 当不可用时，绘制半透明遮罩
        if (!available) {
            Graphics2D g2d = (Graphics2D) g.create();
            g2d.setColor(new Color(0, 0, 0, 150));
            g2d.fillRect(0, 0, getWidth(), getHeight());
            g2d.dispose();
        }

        // 选中时绘制黄色边框
        if (selected) {
            g.setColor(Color.YELLOW);
            g.drawRect(0, 0, getWidth() - 1, getHeight() - 1);
        }
    }
}
