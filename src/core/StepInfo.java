
package core;

public class StepInfo {
    public float[] newState;
    public int[] newAction;
    public double reward;
    public boolean done;
    public boolean truncated;
    public float[] final_obs;

    public StepInfo(float[] state, int[] action, double reward, boolean done, boolean truncated, float[] final_obs) {
        this.newState = state;
        this.newAction = action;
        this.reward = reward;
        this.done = done;
        this.truncated = truncated;

        this.final_obs = final_obs;
    }

    
}
