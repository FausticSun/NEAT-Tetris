import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Parameters {
    private static final Logger LOGGER = Logger.getLogger( Parameters.class.getName() );
    // Experiment parameters
    public String EXPERIMENT_TYPE = "";
    public int GENERATION_LIMIT = 20; // Number of iterations
    public double FITNESS_LIMIT = 1000; // Value for which we automatically end the search

    // Neural Net related parameters
    public int INPUT_SIZE = 2;
    public int OUTPUT_SIZE = 1;
    public int DEFAULT_HIDDEN_SIZE = 0;
    public int DEFAULT_NETWORK_SIZE = 1 + INPUT_SIZE + OUTPUT_SIZE + DEFAULT_HIDDEN_SIZE;
    public int BIAS_START_INDEX = 0;
    public int INPUT_START_INDEX = 1;
    public int OUTPUT_START_INDEX = INPUT_START_INDEX + INPUT_SIZE;
    public int HIDDEN_START_INDEX = OUTPUT_START_INDEX + OUTPUT_SIZE;

    // NEAT parameters
    // Population parameters
    public int POPULATION_SIZE = 100; // Population Size
    public double SURVIVAL_THRESHOLD = 0.2; // Percentage of species allowed to survive and breed
    public int MAXIMUM_POPULATION_STAGNATION = 20; // Generations of non-improvement before population is reduced
    public int TARGET_SPECIES = 10; // No. of species to target using dynamic thresholding
    public double COMPAT_MOD = 0.3; // Amount to tweak compatibility threshold by
    public double COMPATIBILITY_THRESHOLD = 1; // Starting threshold for measuring species compatibility
    // Species parameters
    public double MAXIMUM_SPECIES_STAGNATION = 15; // Generations of non-improvement before species is removed
    public double CROSSOVER_CHANCE = 0.05; // Chance of interspecies breeding
    public double DISJOINT_COEFFICIENT = 1; //  Importance of disjoint genes in measuring compatibility
    public double EXCESS_COEFFICIENT = 1; // Coefficient for excess genes
    public double WEIGHT_DIFFERENCE_COEFFICIENT = 3; // Coefficient for average weight difference
    // Breeding/Mutation parameters
    public double SIMILAR_FITNESS_DISCREPANCY = 0; // Amount of discrepancy for 2 chromosomes to have similar fitness
    public double WEIGHT_MUTATION_RANGE = 2.5; // Range at which the weight can be increased or decreased by
    public double WEIGHT_MUTATION_CHANCE = 0.25; // Chance of weight of gene being changed
    public double NODE_MUTATION_CHANCE = 0.30; // Chance of inserting a new node
    public double LINK_MUTATION_CHANCE = 0.25; // Chance of inserting a new link
    public double DISABLE_MUTATION_CHANCE = 0.04; // Chance of a gene being disabled
    public double ENABLE_MUTATION_CHANCE = 0.02; // Chance of a gene being enabled

    public List<Link> DEFAULT_CHROMOSOME_BLUEPRINT;
    public Function<NeuralNet, Double> FITNESS_EVALUATOR;
    public ExecutorService EXECUTOR = Executors.newWorkStealingPool();

    public static Parameters createXORParameters() {
        Parameters params = new Parameters();
        params.EXPERIMENT_TYPE = "XOR";
        params.DEFAULT_CHROMOSOME_BLUEPRINT = new ArrayList<>();
        params.FITNESS_EVALUATOR = (nn) -> {
            Double[][][] concepts = {
                    {{0.0, 0.0}, {0.0}},
                    {{0.0, 1.0}, {1.0}},
                    {{1.0, 0.0}, {1.0}},
                    {{1.0, 1.0}, {0.0}}
            };
            double error = 0;

            Double[] outputs;
            for (Double[][] c: concepts) {
                outputs = nn.activate(Arrays.asList(c[0])).toArray(new Double[1]);
                error += Math.pow(c[1][0] - outputs[0], 2);
            }

            return 1.0 - error;
        };
        return params;
    }

    public static Parameters createTetrisParameters() {
        Parameters params = new Parameters();
        // Experiment parameters
        params.EXPERIMENT_TYPE = "TETRIS";
        params.GENERATION_LIMIT = 1000; // Number of iterations
        params.FITNESS_LIMIT = 1000; // Value for which we automatically end the search
        // NerualNet parameters
        params.setNNSize(State.ROWS*State.COLS+State.N_PIECES, 4+State.COLS, 1);

        // NEAT parameters
        // Population parameters
        params.POPULATION_SIZE = 250; // Population Size
        params.SURVIVAL_THRESHOLD = 0.2; // Percentage of species allowed to survive and breed
        params.MAXIMUM_POPULATION_STAGNATION = 20; // Generations of non-improvement before population is reduced
        params.TARGET_SPECIES = 10; // No. of species to target using dynamic thresholding
        params.COMPATIBILITY_THRESHOLD = -10; // Starting threshold for measuring species compatibility
        params.COMPAT_MOD = 0.1; // Amount to tweak compatibility threshold by
        // Species parameters
        params.MAXIMUM_SPECIES_STAGNATION = 30; // Generations of non-improvement before species is removed
        params.CROSSOVER_CHANCE = 0.05; // Chance of interspecies breeding
        params.DISJOINT_COEFFICIENT = 1; //  Importance of disjoint genes in measuring compatibility
        params.EXCESS_COEFFICIENT = 1; // Coefficient for excess genes
        params.WEIGHT_DIFFERENCE_COEFFICIENT = 0.8; // Coefficient for average weight difference
        // Breeding/Mutation parameters
        params.SIMILAR_FITNESS_DISCREPANCY = 0.005; // Amount of discrepancy for 2 chromosomes to have similar fitness
        params.WEIGHT_MUTATION_RANGE = 2.5; // Range at which the weight can be increased or decreased by
        params.WEIGHT_MUTATION_CHANCE = 0.1; // Chance of weight of gene being changed
        params.NODE_MUTATION_CHANCE = 0.2; // Chance of inserting a new node
        params.LINK_MUTATION_CHANCE = 0.1; // Chance of inserting a new link
        params.DISABLE_MUTATION_CHANCE = 0.1; // Chance of a gene being disabled
        params.ENABLE_MUTATION_CHANCE = 0.1; // Chance of a gene being enabled

        params.DEFAULT_CHROMOSOME_BLUEPRINT = new ArrayList<>();
        int ONE_HOT_START_INDEX = params.INPUT_START_INDEX + State.ROWS*State.COLS;
        for (int from=ONE_HOT_START_INDEX; from<params.OUTPUT_START_INDEX; from++) {
            params.DEFAULT_CHROMOSOME_BLUEPRINT.add(new Link(from, params.HIDDEN_START_INDEX));
        }
        for (int to=params.OUTPUT_START_INDEX; to<params.HIDDEN_START_INDEX; to++) {
            params.DEFAULT_CHROMOSOME_BLUEPRINT.add(new Link(params.HIDDEN_START_INDEX, to));
        }

        int FITNESS_EVALUATIONS = 500;
        int EVALUATION_PER_THREAD = 100;
        params.FITNESS_EVALUATOR = (nn) -> {
            LOGGER.fine(String.format("Evaluating fitness for C%d", nn.getChromosome().getId()));
            Function<List<Integer>, List<Double>> tetrisFitnessEvaluator = l -> {
                NeuralNet subNn = new NeuralNet(nn);
                return l.stream()
                        .map(i -> {
                            TetrisState s = new TetrisState();
                            while (!s.hasLost()) {
                                s.setOutputs(subNn.activate(s.getInputs()));
                            }
                            return s.getFitness();
                        })
                        .collect(Collectors.toList());
            };
            Function<Integer, Integer> partitioner = i -> i % EVALUATION_PER_THREAD;
            double finalFitness = IntStream.range(0, FITNESS_EVALUATIONS).boxed()
                    .collect(Collectors.groupingBy(partitioner))
                    .values().parallelStream()
                    .map(tetrisFitnessEvaluator)
                    .flatMap(List::stream).collect(Collectors.toList())
                    .stream().mapToDouble(d->d).average().orElse(0);
            return finalFitness;
        };
        return params;
    }

    private void setNNSize(int inputSize, int outputSize, int defaultHiddenSize) {
        INPUT_SIZE = inputSize;
        OUTPUT_SIZE = outputSize;
        DEFAULT_HIDDEN_SIZE = defaultHiddenSize;
        DEFAULT_NETWORK_SIZE = 1 + INPUT_SIZE + OUTPUT_SIZE + DEFAULT_HIDDEN_SIZE;
        BIAS_START_INDEX = 0;
        INPUT_START_INDEX = 1;
        OUTPUT_START_INDEX = INPUT_START_INDEX + INPUT_SIZE;
        HIDDEN_START_INDEX = OUTPUT_START_INDEX + OUTPUT_SIZE;
    }
}
