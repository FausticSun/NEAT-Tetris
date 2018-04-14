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
    public int POPULATION_SIZE = 15; // Population Size
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
    public boolean CLASSIC_TOPOLOGY_MUTATION = false; // Whether to use classic neat mutation or ANJI style mutation
    public double SIMILAR_FITNESS_DISCREPANCY = 0; // Amount of discrepancy for 2 chromosomes to have similar fitness
    public double WEIGHT_MUTATION_RANGE = 1.0; // Range at which the weight can be increased or decreased by
    public double WEIGHT_MUTATION_CHANCE = 0.01; // Chance of weight of gene being changed
    public double NODE_MUTATION_CHANCE = 0.01; // Chance of inserting a new node
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
        int features = 26;
        params.EXPERIMENT_TYPE = "TETRIS";
        params.GENERATION_LIMIT = 10000; // Number of iterations
        params.FITNESS_LIMIT = 1000000; // Value for which we automatically end the search
        // NerualNet parameters
        params.INPUT_SIZE = features+State.ROWS*State.COLS+State.N_PIECES;
        params.OUTPUT_SIZE = 1;
        params.DEFAULT_HIDDEN_SIZE = 0;
        params.setNNSize(params.INPUT_SIZE, params.OUTPUT_SIZE, params.DEFAULT_HIDDEN_SIZE);

        // NEAT parameters
        // Population parameters
        params.POPULATION_SIZE = 50; // Population Size
        params.SURVIVAL_THRESHOLD = 0.2; // Percentage of species allowed to survive and breed
        params.MAXIMUM_POPULATION_STAGNATION = 1000; // Generations of non-improvement before population is reduced
        params.TARGET_SPECIES = 3; // No. of species to target using dynamic thresholding
        params.COMPATIBILITY_THRESHOLD = -10; // Starting threshold for measuring species compatibility
        params.COMPAT_MOD = 0.1; // Amount to tweak compatibility threshold by
        // Species parameters
        params.MAXIMUM_SPECIES_STAGNATION = 1; // Generations of non-improvement before species is removed
        params.CROSSOVER_CHANCE = 0.05; // Chance of interspecies breeding
        params.DISJOINT_COEFFICIENT = 1; //  Importance of disjoint genes in measuring compatibility
        params.EXCESS_COEFFICIENT = 1; // Coefficient for excess genes
        params.WEIGHT_DIFFERENCE_COEFFICIENT = 0.5; // Coefficient for average weight difference
        // Breeding/Mutation parameters
        params.SIMILAR_FITNESS_DISCREPANCY = 0; // Amount of discrepancy for 2 chromosomes to have similar fitness
        params.WEIGHT_MUTATION_RANGE = 1; // Range at which the weight can be increased or decreased by
        params.WEIGHT_MUTATION_CHANCE = 0.01; // Chance of weight of gene being changed
        params.NODE_MUTATION_CHANCE = 0.005; // Chance of inserting a new node
        params.LINK_MUTATION_CHANCE = 0.004; // Chance of inserting a new link
        params.DISABLE_MUTATION_CHANCE = 0.001; // Chance of a gene being disabled
        params.ENABLE_MUTATION_CHANCE = 0.001; // Chance of a gene being enabled

        params.DEFAULT_CHROMOSOME_BLUEPRINT = new ArrayList<>();
        // All bias and el-tetris feature inputs to outputs
        for (int from=0; from<1+6; from++) {
            for (int to=params.OUTPUT_START_INDEX; to<params.HIDDEN_START_INDEX; to++) {
                params.DEFAULT_CHROMOSOME_BLUEPRINT.add(new Link(from, to));
            }
        }
        // All inputs to 1 hidden node to outputs
//        params.DEFAULT_HIDDEN_SIZE = 1;
//        params.setNNSize(params.INPUT_SIZE, params.OUTPUT_SIZE, params.DEFAULT_HIDDEN_SIZE);
//        int ONE_HOT_START_INDEX = params.INPUT_START_INDEX + State.ROWS*State.COLS;
//        for (int from=ONE_HOT_START_INDEX; from<params.OUTPUT_START_INDEX; from++) {
//            params.DEFAULT_CHROMOSOME_BLUEPRINT.add(new Link(from, params.HIDDEN_START_INDEX));
//        }
//        for (int to=params.OUTPUT_START_INDEX; to<params.HIDDEN_START_INDEX; to++) {
//            params.DEFAULT_CHROMOSOME_BLUEPRINT.add(new Link(params.HIDDEN_START_INDEX, to));
//        }

        // Only piece type inputs to outputs
//        params.DEFAULT_HIDDEN_SIZE = 0;
//        params.setNNSize(params.INPUT_SIZE, params.OUTPUT_SIZE, params.DEFAULT_HIDDEN_SIZE);
//        int ONE_HOT_START_INDEX = params.INPUT_START_INDEX + State.ROWS*State.COLS;
//        for (int from=ONE_HOT_START_INDEX; from<params.OUTPUT_START_INDEX; from++) {
//            for (int to=params.OUTPUT_START_INDEX; to<params.HIDDEN_START_INDEX; to++) {
//                params.DEFAULT_CHROMOSOME_BLUEPRINT.add(new Link(from, to));
//            }
//        }

        // All inputs to outputs
//        params.DEFAULT_HIDDEN_SIZE = 0;
//        params.setNNSize(params.INPUT_SIZE, params.OUTPUT_SIZE, params.DEFAULT_HIDDEN_SIZE);
//        for (int from=params.BIAS_START_INDEX; from<params.OUTPUT_START_INDEX; from++) {
//            for (int to=params.OUTPUT_START_INDEX; to<params.HIDDEN_START_INDEX; to++) {
//                params.DEFAULT_CHROMOSOME_BLUEPRINT.add(new Link(from, to));
//            }
//        }

        // Piece type inputs connects directly to outputs, other inputs connect via 1 hidden node
//        params.DEFAULT_HIDDEN_SIZE = 1;
//        params.setNNSize(params.INPUT_SIZE, params.OUTPUT_SIZE, params.DEFAULT_HIDDEN_SIZE);
//        int ONE_HOT_START_INDEX = params.INPUT_START_INDEX + State.ROWS*State.COLS;
//        for (int from=ONE_HOT_START_INDEX; from<params.OUTPUT_START_INDEX; from++) {
//            for (int to=params.OUTPUT_START_INDEX; to<params.HIDDEN_START_INDEX; to++) {
//                params.DEFAULT_CHROMOSOME_BLUEPRINT.add(new Link(from, to));
//            }
//        }
//        for (int from=params.INPUT_START_INDEX; from<ONE_HOT_START_INDEX; from++) {
//            params.DEFAULT_CHROMOSOME_BLUEPRINT.add(new Link(from, params.HIDDEN_START_INDEX));
//        }
//        for (int to=params.OUTPUT_START_INDEX; to<params.HIDDEN_START_INDEX; to++) {
//            params.DEFAULT_CHROMOSOME_BLUEPRINT.add(new Link(params.HIDDEN_START_INDEX, to));
//        }

        // Piece type inputs connects directly to outputs, other inputs connect via 5 hidden nodes
//        params.DEFAULT_HIDDEN_SIZE = 5;
//        params.setNNSize(params.INPUT_SIZE, params.OUTPUT_SIZE, params.DEFAULT_HIDDEN_SIZE);
//        int ONE_HOT_START_INDEX = params.INPUT_START_INDEX + State.ROWS*State.COLS;
//        for (int from=ONE_HOT_START_INDEX; from<params.OUTPUT_START_INDEX; from++) {
//            for (int to=params.OUTPUT_START_INDEX; to<params.HIDDEN_START_INDEX; to++) {
//                params.DEFAULT_CHROMOSOME_BLUEPRINT.add(new Link(from, to));
//            }
//        }
//        for (int from=params.INPUT_START_INDEX; from<ONE_HOT_START_INDEX; from++) {
//            for (int to=params.HIDDEN_START_INDEX; to<params.HIDDEN_START_INDEX+params.DEFAULT_HIDDEN_SIZE; to++) {
//                params.DEFAULT_CHROMOSOME_BLUEPRINT.add(new Link(from, to));
//            }
//        }
//        for (int from=params.HIDDEN_START_INDEX; from<params.HIDDEN_START_INDEX+params.DEFAULT_HIDDEN_SIZE; from++) {
//            for (int to=params.OUTPUT_START_INDEX; to<params.HIDDEN_START_INDEX; to++) {
//                params.DEFAULT_CHROMOSOME_BLUEPRINT.add(new Link(from, to));
//            }
//        }

        int FITNESS_EVALUATIONS = 1;
        params.FITNESS_EVALUATOR = (nn) -> {
            LOGGER.fine(String.format("Evaluating fitness for C%d", nn.getChromosome().getId()));
            double finalFitness = IntStream.range(0, FITNESS_EVALUATIONS)
                    .mapToDouble(i -> {
                        TetrisState s = new TetrisState(nn);
//                        TFrame demo = new TFrame(s);

                        while (!s.hasLost()) {
                            s.makeBestMove();

//                            s.draw();
//                            s.drawNext(0, 0);
                        }
//                        demo.dispose();
                        return s.getFitness();
                    })
                    .average().orElse(0);
            LOGGER.fine(String.format("C%d fitness: %f", nn.getChromosome().getId(), finalFitness));
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
