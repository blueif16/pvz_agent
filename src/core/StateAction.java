package core;


public class StateAction {
    private float[] state;
    private int[] action;

    public StateAction(float[] state, int[] action) {
        this.state = state;
        this.action = action;
    }

    public float[] getState(){
        return state;
    }

    public int[] getAction(){
        return action;
    }

}
