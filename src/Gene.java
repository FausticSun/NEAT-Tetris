import java.util.Random;

public class Gene extends Innovation {
    private Parameters params;
    private double weight = 0;
    private boolean isEnabled;

    public Gene(Parameters params, Innovation innovation) {
        super(innovation);
        this.params = params;
        perterbWeight();
        isEnabled = true;
    }

    public Gene(Gene o) {
        super(o);
        this.params = o.params;
        this.weight = o.weight;
        isEnabled = true;
        this.isEnabled = o.isEnabled;
    }

    public void perterbWeight() {
        this.weight += (new Random()).nextGaussian()*params.WEIGHT_MUTATION_RANGE;
    }

    public double getWeight() {
        return weight;
    }

    public boolean isEnabled() {
        return isEnabled;
    }

    public void toggleEnabled() {
        isEnabled = !isEnabled;
    }

    public String toString() {
        return String.format("%d, %d, %d, %f, %b", getId(), getFrom(), getTo(), getWeight(), isEnabled());
    }
}
