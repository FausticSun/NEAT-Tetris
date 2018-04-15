import java.util.*;
import java.util.logging.Logger;

public class NeuralNet {
    private static final Logger LOGGER = Logger.getLogger( NeuralNet.class.getName() );
    private Parameters params;
    private Map<Integer, Neuron> neurons = new HashMap<>();
    private Chromosome chromosome;

    public NeuralNet(Parameters params, Chromosome chromosome) {
        this.params = params;
        this.chromosome = chromosome;
        // Initialize bias neuron
        neurons.put(params.BIAS_START_INDEX, new Neuron(params.BIAS_START_INDEX, ActivationType.BIAS));
        // Initialize input neurons
        for (int i=params.INPUT_START_INDEX; i<params.OUTPUT_START_INDEX; i++) {
            neurons.put(i, new Neuron(i, ActivationType.LINEAR));
        }
        // Initialize output neurons
        for (int i=params.OUTPUT_START_INDEX; i<params.HIDDEN_START_INDEX; i++) {
            neurons.put(i, new Neuron(i, ActivationType.SIGMOID));
        }
        // Add all links and required hidden neurons
        for (Gene g: chromosome.getGenes()) {
            if (g.isEnabled()) {
                getNeuron(g.getTo()).addLink(getNeuron(g.getFrom()), g.getWeight());
            }
        }
    }

    public NeuralNet(NeuralNet other) {
        this(other.params, other.chromosome);
    }

    private Neuron getNeuron(int id) {
        if (!neurons.containsKey(id)) {
            neurons.put(id, new Neuron(id, ActivationType.SIGMOID));
        }
        return neurons.get(id);
    }

    public List<Double> activate(List<Double> inputs) {
        // Check input size
        if (inputs.size() != params.INPUT_SIZE) {
            LOGGER.severe(String.format("Input size mismatch in NN%d, %d supplied when %d required",
                    chromosome.getId(),
                    inputs.size(),
                    params.INPUT_SIZE));
            return null;
        }

        // Clear the network
        reset();

        // Set Input Neurons
        for (int i=1; i<params.OUTPUT_START_INDEX; i++) {
            neurons.get(i).setValue(inputs.get(i-1));
        }

        // Activate Output Neurons
        List<Double> output = new ArrayList<>();
        for (int i=params.OUTPUT_START_INDEX; i<params.HIDDEN_START_INDEX; i++) {
            output.add(neurons.get(i).getValue());
        }

        // Return output
        return output;
    }

    public void reset() {
        for (Neuron n: neurons.values()) {
            n.reset();
        }
    }

    public Chromosome getChromosome() {
        return chromosome;
    }

    class Neuron {
        private int id;
        private List<NeuronLink> incomingLinks = new ArrayList<>();
        private ActivationType activationType;
        private double value;
        private boolean isActive = false;

        public Neuron(int id, ActivationType activationType) {
            this.id = id;
            this.activationType = activationType;
        }

        public void addLink(Neuron neuron, double weight) {
            incomingLinks.add(new NeuronLink(neuron, weight));
        }

        public void activate() {
            double sum = 0;
            for (NeuronLink link: incomingLinks) {
                sum += link.getWeightedValue();
            }
            switch (this.activationType) {
                case BIAS: this.value = 1; break;
                case LINEAR: this.value = sum; break;
                case SIGMOID: this.value = sigmoid(sum); break;
            }
            this.isActive = true;
        }

        private double sigmoid(double x) {
            return 1 / (1 + Math.exp(-x));
        }

        public void setValue(double value) {
            this.value = value;
            this.isActive = true;
        }

        public double getValue() {
            if (!this.isActive)
                this.activate();
            return this.value;
        }

        public void reset() {
            this.value = 0;
            this.isActive = false;
        }

        public int getId() {
            return id;
        }
    }

    class NeuronLink {
        private Neuron outgoingNeuron;
        private double weight;

        public NeuronLink(Neuron neuron, double weight) {
            this.outgoingNeuron = neuron;
            this.weight = weight;
        }

        public double getWeightedValue() {
            return outgoingNeuron.getValue() * weight;
        }
    }

    enum ActivationType {
        BIAS, LINEAR, SIGMOID
    }
}
