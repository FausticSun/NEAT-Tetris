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
        this.weight += Math.random() * params.WEIGHT_MUTATION_RANGE*2 - params.WEIGHT_MUTATION_RANGE;
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
}
