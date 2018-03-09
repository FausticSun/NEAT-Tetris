import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;
import java.util.logging.Logger;
import java.lang.StringBuilder;

public class PlayerSkeleton {
    private static final Logger LOGGER = Logger.getLogger( PlayerSkeleton.class.getName() );
	static boolean HEADLESS = false;
	static boolean EVOLVE = false;

	public static void main(String[] args) {
		for (String arg : args) {
			switch(arg) {
				case "headless":
					HEADLESS = true; break;
				case "evolve":
					EVOLVE = true; break;
				default:
					break;
			}
		}
		new PlayerSkeleton();
	}

	public PlayerSkeleton() {
        Experiment ex = new TetrisExperiment();
        ex.run();
        Chromosome fittest = ex.getFittest();
        if (ex instanceof TetrisExperiment) {
            State s = new State();
            NeuralNet nn = new NeuralNet(fittest);

            if (!HEADLESS)
                new TFrame(s);
            while (!s.hasLost()) {
                nn.activate(StateUtils.normalize(s));
                List<Double> output = nn.getOutput();

                int orient = StateUtils.getOrient(s, output);
                int slot = StateUtils.getSlot(s, orient, output);
                if (slot == -1) {
                    s.lost = true;
                    continue;
                }

                s.makeMove(orient, slot);

                if (!HEADLESS) {
                    s.draw();
                    s.drawNext(0, 0);
                    try {
                        Thread.sleep(300);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
            LOGGER.info(String.format("%d rows cleared", s.getRowsCleared()));
        }
    }
}

class Species {
    public static double CROSSOVER_CHANCE;
    public static double COMPATIBILITY_THRESHOLD;
    public static double SURVIVAL_THRESHOLD;
    private Population pop;
	private Chromosome representative;
	private List<Chromosome> chromosomes;
	private int stagnation;
	private double bestFitness;
	private double averageFitness;
	private double allocatedOffsprings;
	private int id;

    /**
     * @param pop The population the species belong to
     * @param representative The species representative
     */
	public Species(Population pop, Chromosome representative){
	    this.pop = pop;
		this.chromosomes = new ArrayList<>();
		this.representative = representative;
		this.chromosomes.add(representative);
		this.id = pop.getNewSpeciesId();
		this.stagnation = 0;
		this.averageFitness = -1;
		this.allocatedOffsprings = -1;
	}

	/**
	 * Slaughter chromosomes below survival threshold
	 */
	public void cull() {
		Collections.sort(chromosomes, Collections.reverseOrder());
		int limit = (int) Math.ceil(chromosomes.size() * SURVIVAL_THRESHOLD);
		chromosomes = chromosomes.subList(0, limit);
	}

    /**
     * Clear all chromosomes from the species and other variables
     */
	public void clear() {
        chromosomes.clear();
        averageFitness = -1;
        allocatedOffsprings = -1;
    }

    /**
     * @return No. of chomosomes in the species
     */
	public int size() {
	    return chromosomes.size();
    }

	/**
	 *
	 * @return returns the average fitness of the chromosomes in the species
	 */
	public double computeAverageFitness() {
		if (chromosomes.size() == 0)
			return averageFitness = 0;
		return averageFitness = chromosomes.stream()
                .mapToDouble(Chromosome::getFitness)
                .sum() / chromosomes.size();
	}

	/**
	 * creates baby chromosomes equal to the amount allocated
	 * if there is only 1 parent, it will always crossbreed
	 * otherwise, it will always breed with a different parent
	 * @return returns a list of new baby chromosomes
	 */
	public List<Chromosome> produceAllocatedOffsprings() {
		List<Chromosome> newChildren = new ArrayList<>();
		if (chromosomes.isEmpty())
		    return newChildren;
		newChildren.add(Collections.max(chromosomes)); // Add species champion
		Chromosome parent1, parent2;
		while(newChildren.size() < allocatedOffsprings) {
			parent1 = getRandomChromosome();
			if (Math.random() < CROSSOVER_CHANCE)
                parent2 = pop.getRandomChromosomeFromSpecies();
			else
			    parent2 = getRandomChromosome();
			newChildren.add(parent1.breedWith(parent2));
		}
		return newChildren;
	}

    /**
     * Check if chromosome is compatible with this species
     * @param c Chromosome to check
     * @return True if compatible, False otherwise
     */
	public boolean compatibleWith(Chromosome c) {
        return representative.distanceFrom(c) < COMPATIBILITY_THRESHOLD;
    }

    /**
     * Add specified chromsome to the species
     * @param c Chromosome to add
     */
    public void add(Chromosome c) {
	    chromosomes.add(c);
    }

    /**
     * @return A random chromosome from this species
     */
	public Chromosome getRandomChromosome() {
	    if (chromosomes.size() == 0)
	        return null;
        return chromosomes.get((new Random()).nextInt(chromosomes.size()));
    }

    /**
     * Confirm that all chromosomes have been added and
     * perform relevant computations
     */
    public void confirmSpecies() {
        // Check for fitness improvement
        double newBestFitness = Collections.max(chromosomes).getFitness();
        if (bestFitness > newBestFitness) {
            stagnation++;
        } else {
            bestFitness = newBestFitness;
            stagnation = 0;
        }
        // Compute average fitness
        averageFitness = computeAverageFitness();
    }

    public double getAverageFitness() {
        return averageFitness;
    }

    public void setAllocatedOffsprings(int allocatedOffsprings) {
        this.allocatedOffsprings = allocatedOffsprings;
    }

    public int getId() {
        return id;
    }
}

/**
 * Feed-forward neural network
 * Neurons are arranged in the List from Input, Output and Hidden
 */
class NeuralNet {
    private static final Logger LOGGER = Logger.getLogger( NeuralNet.class.getName() );
    public static int INPUT_SIZE;
    public static int OUTPUT_SIZE;
    public static int BIAS_START_INDEX;
    public static int INPUT_START_INDEX;
    public static int OUTPUT_START_INDEX;
    public static int HIDDEN_START_INDEX;
	private List<Neuron> neurons;
    private Chromosome chromosome;

	/**
	 * A neural net built based on a specific chromosome data.
	 * Call activate() to pass in the input and then getOuput() later.
	 *
	 * @param chromosome The chromosome to create the neural net information.
	 */
	public NeuralNet(Chromosome chromosome) {
		this.chromosome = chromosome;
        LOGGER.fine(String.format("Creating neural network for C%d of size %d",
                chromosome.getId(), chromosome.getNeuronCount()));
		// Create Neurons
		neurons = new ArrayList<>();
        neurons.add(new Neuron(ActivationType.BIAS)); // Bias Neurons
        for (int i=0; i<INPUT_SIZE; i++)
            neurons.add(new Neuron(ActivationType.LINEAR)); // Input Neurons
        for (int i=0; i<chromosome.getNeuronCount()-INPUT_SIZE-1; i++)
            neurons.add(new Neuron(ActivationType.SIGMOID)); // Output and Hidden Neurons

		// Insert links
		for (Gene g: chromosome.getGenes()) {
			if (g.isEnabled) {
				neurons.get(g.to).addLink(neurons.get(g.from), g.weight);
			}
		}
	}
	
	/**
	 * Places these inputs into the neural net input neurons.
	 *
	 * @param inputs The list of inputs from the screen and the current piece.
	 * @return The list of outputs, null if unsuccessful
	 */
	public List<Double> activate(List<Double> inputs) {
		// Check input size
		if (inputs.size() != INPUT_SIZE) {
			LOGGER.warning("Input size mismatch!");
			return null;
		}

		// Clear the network
		reset();

		// Set Input Neurons
		for (int i=0; i<INPUT_SIZE; i++) {
			neurons.get(i).setValue(inputs.get(i));
		}

		// Activate Output Neurons
        List<Double> output = new ArrayList<>();
		for (int i=OUTPUT_START_INDEX; i<HIDDEN_START_INDEX; i++) {
			output.add(neurons.get(i).getValue());
		}

		// Return output
        return output;
	}
	
	/**
	 * Gets the output of the neural net.
	 * Call this after you call activate and pass in the inputs.
	 *
	 * @return The list of outputs from the chromosome.
	 */
	public List<Double> getOutput() {
		List<Double> output = new ArrayList<Double>();
		for (int i=OUTPUT_START_INDEX; i<HIDDEN_START_INDEX; i++) {
			output.add(neurons.get(i).value);
		}
		return output;
	}

	public void reset() {
		for (int i=OUTPUT_START_INDEX; i<neurons.size(); i++)
			neurons.get(i).reset();
	}

    /**
     * Neuron refers to a specific node, be in input, output, hidden or bias.
     * This stores information about the node and incoming links.
     */
    class Neuron {
        private List<NeuronLink> incomingLinks;
        private ActivationType type;
        private double value;
        private boolean isActive;

        public Neuron(ActivationType type) {
            this.incomingLinks = new ArrayList<>();
            this.type = type;
            this.value = 0;
            this.isActive = false;
        }

        /**
         * Recursively activate dependent neurons
         */
        public Neuron activate() {
            double sum = 0;
            for (NeuronLink link: incomingLinks) {
                sum += link.getWeightedValue();
            }
            switch (this.type) {
                case BIAS: this.value = 1;
                case LINEAR: this.value = sum; break;
                case SIGMOID: this.value = Calc.sigmoid(sum); break;
            }
            this.isActive = true;
            return this;
        }

        public Neuron addLink(Neuron neuron, Double weight) {
            incomingLinks.add(new NeuronLink(neuron, weight));
            return this;
        }

        public Neuron setValue(double value) {
            this.value = value;
            this.isActive = true;
            return this;
        }

        public double getValue() {
            if (!this.isActive)
                this.activate();
            return this.value;
        }

        public Neuron reset() {
            this.value = 0;
            this.isActive = false;
            return this;
        }

        /**
         * Represents an incoming neuron link.
         */
        class NeuronLink {
            private Neuron neuron;
            private double weight;

            /**
             * @param neuron The id of the incoming neuron
             * @param weight The weight of this link
             */
            public NeuronLink(Neuron neuron, double weight) {
                this.neuron = neuron;
                this.weight = weight;
            }

            public double getWeightedValue() {
                return neuron.getValue()*weight;
            }
        }
    }

    enum ActivationType {
        LINEAR, SIGMOID, BIAS
    }
}

class Calc {
	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}
}

class Chromosome implements Comparable<Chromosome> {
    private static final Logger LOGGER = Logger.getLogger( Chromosome.class.getName() );
    public static double WEIGHT_MUTATION_CHANCE;
    public static double NODE_MUTATION_CHANCE;
    public static double LINK_MUTATION_CHANCE;
    public static double ENABLE_MUTATION_CHANCE;
    public static double DISABLE_MUTATION_CHANCE;
    public static double DISJOINT_COEFFICIENT;
    public static double EXCESS_COEFFICIENT;
    public static double WEIGHT_DIFFERENCE_COEFFICIENT;

    // Array indices for stucturalDiff
    private static final int SAME = 0;
    private static final int DISJOINT = 1;
    private static final int EXCESS = 2;

	private Population pop;
    private int neuronCount;
	private List<Gene> genes;
	private double fitness;
	private int id;

    /**
     * Initializes a default chromosome. This should only be called once throughout the experiment.
     * @param pop The population this chromosome belongs to
     * @param genes The initial genes in this chromosome
     * @param neuronCount The maximum neuron id referenced by any gene
     */
	public Chromosome(Population pop, List<Gene> genes, int neuronCount) {
		this.pop = pop;
        this.genes = genes;
        this.neuronCount = neuronCount;
		this.fitness = -1;
		this.id = -1;
        LOGGER.info(String.format("Base gene size %d", genes.size()));
    }

    /**
     * Clones a chromosome. This should be the only way new chromosomes are created in the experiment.
     * @param other The chromosome to clone
     */
	public Chromosome(Chromosome other) {
	    this.pop = other.pop;
		this.neuronCount = other.neuronCount;
		this.genes = new ArrayList<>();
		for (Gene g : other.genes)
			this.genes.add(new Gene(g));
		this.fitness = other.fitness;
		this.id = this.pop.getNewChromosomeId();
		LOGGER.finest(String.format("Cloning C%d to C%d",
                other.id,
                this.id));
	}

	/**
     * Compare Chromosomes based on their fitness
	 * @param other - Other chromosome being tested against
	 * @return Positive if better, negative if worse
	 */
	@Override
	public int compareTo(Chromosome other) {
	    return Double.compare(this.fitness, other.fitness);
	}

	/**
	 * Breeds 2 chromosomes together:
	 * Same ID genes get randomly picked from one parent
	 * Excess and Disjoint genes get picked from the better parent
	 * If both parents are the same, Excess and Disjoint genes get added from both
	 *
	 * @param other - other chromosome being bred with
	 * @return the baby chromosome
	 */
	public Chromosome breedWith(Chromosome other) {
		// Ensure that this has a higher fitness than other.
		if(other.fitness > this.fitness)
			return other.breedWith(this);

        LOGGER.finer(String.format("Breeding C%d with C%d", this.id, other.id));
        // Compare the parents
        int[] structuralDifferences = calculateStructuralDifferences(other);

		// Clone child from fitter parent
		Chromosome child = new Chromosome(this);

		// Randomly replace same genes from weaker parent
        Collections.sort(child.genes);
        Collections.sort(other.genes);
        for (int i=0; i<structuralDifferences[SAME]; i++) {
            if (Math.random() < 0.5) {
                child.genes.remove(i);
                child.genes.add(i, new Gene(other.genes.get(i)));
            }
        }

        // Add excess and disjoint genes from weaker parent if
        // fitness is the same
        if (this.getFitness() == other.getFitness()) {
            for (int i=structuralDifferences[SAME]; i<other.genes.size(); i++) {
                child.genes.add(new Gene(other.genes.get(i)));
            }
            child.neuronCount = Math.max(child.neuronCount, other.neuronCount);
        }

		// Mutate child
        child.mutate();

		return child;
	}

	/**
	 * Mutates the chromosome randomly:
	 * Each gene can mutate weight or get disabled/enabled
	 * Chromosome can gain a new node or link
     * Requires a reevaluation of fitness
	 */
	public Chromosome mutate() {
	    LOGGER.finest(String.format("Mutating C%d", this.id));
	    double randomNumber = Math.random();
	    this.fitness = -1;

		if (randomNumber < LINK_MUTATION_CHANCE) {
            mutateLink();
            return this;
        }
		randomNumber -= LINK_MUTATION_CHANCE;
		if (randomNumber  < NODE_MUTATION_CHANCE) {
            mutateNode();
            return this;
        }
        randomNumber -= NODE_MUTATION_CHANCE;
		if (randomNumber < WEIGHT_MUTATION_CHANCE)
        {
            mutateGeneWeight();
            return this;
        }
        randomNumber -= WEIGHT_MUTATION_CHANCE;

		return this;
	}

	/**
	 * mutates by creating a link
	 * To make it easy for validation for now, only options are
	 * 1) must start from a input node
	 * 2) must end at a output node
	 */
	public void mutateLink() {
	    int from, to;
	    boolean isExist = false;

	    // Get a list of neurons present in this chromosome
	    Set<Integer> presentNeuronsSet = new TreeSet<>();
	    for (Gene g: genes) {
            presentNeuronsSet.add(g.from);
            presentNeuronsSet.add(g.to);
        }
        List<Integer> presentNeuronsList = new ArrayList<>(presentNeuronsSet);

	    // Attempt to get 2 neurons that fulfills all requriements
	    for (int i=0; i<10; i++) {
	        // Get 2 neurons
	        from = presentNeuronsList.get((new Random()).nextInt(presentNeuronsList.size()));
            to = presentNeuronsList.get((new Random()).nextInt(presentNeuronsList.size()));
            // Check if the neurons are the same
            if (from == to)
                continue;
            // Check if the neurons are 2 inputs
            if (from < NeuralNet.OUTPUT_START_INDEX && to < NeuralNet.OUTPUT_START_INDEX)
                continue;
            // Check if the neurons are 2 outputs
            if (from >= NeuralNet.OUTPUT_START_INDEX && from < NeuralNet.HIDDEN_START_INDEX &&
                    to >= NeuralNet.OUTPUT_START_INDEX && from < NeuralNet.HIDDEN_START_INDEX)
                continue;
            // Check if a link already exists
            for (Gene g: genes) {
                if ((g.from == from && g.to == to) ||
                        (g.from == to && g.to == from)) {
                    isExist = true;
                }
            }
            if (isExist)
                continue;
            // If 2 neurons are hidden, perform DFS to determine from and to
            if (to >= NeuralNet.HIDDEN_START_INDEX && from >= NeuralNet.HIDDEN_START_INDEX) {
                if (!dfs(from, to)) {
                    int temp = from;
                    from = to;
                    to = temp;
                }
            }

            // Else, make sure input neuron is from or output neuron is to
            else if (to < NeuralNet.OUTPUT_START_INDEX ||
                    (from >= NeuralNet.OUTPUT_START_INDEX && from < NeuralNet.HIDDEN_START_INDEX)) {
                int temp = from;
                from = to;
                to = temp;
            }

            // Make the link
            Gene newGene = pop.getInnovator().innovateLink(from, to);
            genes.add(newGene);
            break;
        }
	}

	/**
	 * mutates by creating a node
	 * can mutate a currently disabled node
	 */
	public void mutateNode() {
	    Gene chosenGene = genes.get((new Random()).nextInt(genes.size()));
	    chosenGene.isEnabled = false;
        Gene[] newGenes = pop.getInnovator().innovateNode(chosenGene.from, chosenGene.to, chosenGene.weight);
        genes.addAll(Arrays.asList(newGenes));
        neuronCount = Math.max(newGenes[0].to+1, neuronCount);
        LOGGER.finest(String.format("Creating new node N%d between N%d and N%d",
                newGenes[0].to, chosenGene.from, chosenGene.to));
	}


    /**
     * Select a random gene and mutates its weight
     */
	public void mutateGeneWeight() {
        genes.get((new Random()).nextInt(genes.size())).mutateWeight();
    }

    /**
     * Perform a dfs from from to to
     * @param from The node to start from
     * @param to The node to end at
     * @return True if you can reach to from from, False otherwise
     */
    public boolean dfs(int from, int to) {
        Set<Integer> visited = new TreeSet<>();
        Stack<Integer> stack = new Stack<>();
        stack.push(from);
        while (!stack.empty()) {
            final int v = stack.pop();
            if (!visited.contains(v)) {
                visited.add(v);
                for (int i: genes.stream()
                        .filter(g -> g.from == v)
                        .mapToInt(g -> g.to)
                        .toArray()) {
                    if (i == to)
                        return true;
                    stack.push(i);
                };
            }
        }
        return false;
    }

	/**
	 * computes distance between genes compared to another chromosome
	 * used for species placement
	 *
	 * @return Distance from this chromosome to other chromosome
	 */
	public double distanceFrom(Chromosome other) {
		double distance = 0;
		double NormalizeValue = Math.max(genes.size(), other.genes.size());
		int[] structuralDifference = calculateStructuralDifferences(other);
        double averageWeightDifferences = 0;
        for (int i=0; i<structuralDifference[SAME]; i++)
            averageWeightDifferences += Math.abs(this.genes.get(i).weight - other.genes.get(i).weight);
        averageWeightDifferences = averageWeightDifferences / structuralDifference[SAME];
		distance += EXCESS_COEFFICIENT * structuralDifference[EXCESS] / NormalizeValue;
		distance += DISJOINT_COEFFICIENT * structuralDifference[DISJOINT] / NormalizeValue;
		distance += WEIGHT_DIFFERENCE_COEFFICIENT * averageWeightDifferences;
		return distance;
	}

    /**
     * Compare this chromosome with other chromosome and determine
     * the number of same, excess and disjoint genes
     * @param other Chromosome to compare with
     * @return Number of same, excess and disjoint genes
     */
	public int[] calculateStructuralDifferences(Chromosome other) {
	    int[] structuralDiff = new int[3];
        Collections.sort(this.genes);
        Collections.sort(other.genes);
        // Get number of same genes
        Iterator<Gene> thisIt = this.genes.iterator();
        Iterator<Gene> otherIt = other.genes.iterator();
        while (thisIt.hasNext() && otherIt.hasNext()) {
            if(thisIt.next().id == otherIt.next().id)
                structuralDiff[SAME]++;
            else
                break;
        }
        // Get number of excess genes
        int thisMaxId = Collections.max(this.genes).id;
        int otherMaxId = Collections.max(other.genes).id;
        // There are no excess and disjoint
        if (thisMaxId == otherMaxId)
            return structuralDiff;
        int minMaxId = Math.min(thisMaxId, otherMaxId);
        // Compute excess genes
        ListIterator<Gene> listIt;
        if (thisMaxId > otherMaxId) {
            listIt = this.genes.listIterator(this.genes.size());
        } else {
            listIt = other.genes.listIterator(other.genes.size());
        }
        while (listIt.previous().id > minMaxId) {
            structuralDiff[EXCESS]++;
        }
        // Compute disjoint genes
        while (thisIt.hasNext() && thisIt.next().id <= minMaxId)
            structuralDiff[DISJOINT]++;
        while (otherIt.hasNext() && otherIt.next().id <= minMaxId)
            structuralDiff[DISJOINT]++;

        return structuralDiff;
    }

	public double getFitness() {
	    return this.fitness;
    }

    public Chromosome setFitness(double fitness) {
	    this.fitness = fitness;
	    return this;
    }

    public boolean isEvaluated() {
        return this.fitness != -1;
    }

    public int getNeuronCount() {
	    return this.neuronCount;
    }

    public List<Gene> getGenes() {
	    return this.genes;
    }

    public int getId() {
	    return this.id;
    }
}

class Gene implements Comparable<Gene>{
    public static double WEIGHT_MUTATION_RANGE;
    public int id;
    public int from;
    public int to;
    public double weight;
    public boolean isEnabled;

    /**
     * Used in innovator to create a reference gene
     * @param id Innovation ID of gene
     * @param from Neuron ID giving output
     * @param to Neuron ID receiving input
     */
    public Gene(int id, int from, int to) {
        this.id = id;
        this.from = from;
        this.to = to;
        this.weight = 0;
        this.isEnabled = true;
    }

    /**
     * Clones a gene
     * @param other Gene to clone
     */
    public Gene(Gene other) {
        this.id = other.id;
        this.from = other.from;
        this.to = other.to;
        this.weight = other.weight;
        this.isEnabled = other.isEnabled;
    }

    public Gene(int id, int from, int to, double weight) {
        this.id = id;
        this.from = from;
        this.to = to;
        this.weight = weight;
        this.isEnabled = true;
    }

    public int compareTo(Gene other) {
        return id - other.id;
    }

    /**
     * mutates by disabling a link
     */
    public Gene mutateDisable() {
        isEnabled = false;
        return this;
    }

    /**
     * mutates by enabling a link
     */
    public Gene mutateEnable() {
        isEnabled = true;
        return this;
    }

    /**
     * mutates by changing weight
     */
    public Gene mutateWeight() {
        weight += Math.random() * WEIGHT_MUTATION_RANGE*2 - WEIGHT_MUTATION_RANGE;
        return this;
    }
}

class FittestChromosome {
	public String xml;
	public FittestChromosome() {
		StringBuilder sb = new StringBuilder();
		sb.append("");
		xml = sb.toString();
	}

	public FittestChromosome(Chromosome chromosome) {
		StringBuilder sb = new StringBuilder();
		sb.append(chromosome.toString());
		xml = sb.toString();
	}
}

/**
 * Static utility function that may prove useful.
 *
 */
class StateUtils {
	// Maximum legal slots given piece type and orient
	public static int[][] maxSlots = new int[State.N_PIECES][];
	// Initalize maxSlots
	static {
		// For each piece type
		for (int i=0; i<State.N_PIECES; i++) {
			int[][] moves = State.legalMoves[i];
			maxSlots[i] = new int[State.pOrients[i]];
			// For each orientation
			for (int j=0; j<State.pOrients[i]; j++) {
				// Count number of moves in legalMoves
				for (int[] move: moves) {
					if (move[State.ORIENT] == j) {
						maxSlots[i][j]++;
					}
				}
			}
		}
	}

	public static List<Double> normalize(State s) {
		List<Double> inputs = new ArrayList<Double>();

		// Convert field to binary input
		int[][] field = s.getField();
		for (int i=0; i<State.ROWS; i++) {
			for (int j=0; j<State.COLS; j++) {
				inputs.add(field[i][j] == 0 ? 0.0 : 1.0);
			}
		}

		// Convert next piece to one-hot input
		int nextPiece = s.getNextPiece();
		for (int i=0; i<State.N_PIECES; i++) {
			inputs.add(i == nextPiece ? 1.0 : 0.0);
		}

		return inputs;
	}
	
	public static int getOrient(State s, List<Double> outputs) {
		int nextPiece = s.getNextPiece();
		double max = outputs.get(0);
		int orient = 0;
		for (int i=1; i<4; i++) {
			if (outputs.get(i) > max) {
				max = outputs.get(i);
				orient = i;
			}
		}
		switch(nextPiece) {
			case 0:
				return 0;
			case 1:
			case 5:
			case 6:
				return orient/2;
			default:
				return orient;
		}
	}
	
	/**
	 *
	 * @param s
	 * @param orient
	 * @param outputs
	 * @return -1 if dropping in an invalid slot
	 */
	public static int getSlot(State s, int orient, List<Double> outputs) {
		double max = outputs.get(4);
		int slot = 0;
		for (int i=5; i<outputs.size(); i++) {
			if (outputs.get(i) > max) {
				max = outputs.get(i);
				slot = i-4;
			}
		}

		// System.out.printf("%d, %d, %d%n", s.nextPiece, slot, maxSlots[s.nextPiece][orient]);
		return slot < maxSlots[s.nextPiece][orient] ? slot : -1;
	}

	public static double getPercentageFilled(State s) {
	    double noFilled = 0.0;
        // Count no of tiles filled
        int[][] field = s.getField();
        for (int i=0; i<State.ROWS; i++) {
            for (int j=0; j<State.COLS; j++) {
                noFilled += field[i][j] > 0 ? 1 : 0;
            }
        }

        return noFilled / (State.ROWS * State.COLS);
    }
}

/**
 * Handles running an experiment
 */
abstract class Experiment {
    private static final Logger LOGGER = Logger.getLogger( Experiment.class.getName() );
    public static double FITNESS_LIMIT;
    public static int GENERATION_LIMIT;
    protected Population pop;
    protected Parameters params;

    /**
     * Setup the experiment by setting static variables of relevant classes
     */
    protected void setup() {
        Experiment.FITNESS_LIMIT = params.FITNESS_LIMIT;
        Experiment.GENERATION_LIMIT = params.GENERATION_LIMIT;
        NeuralNet.INPUT_SIZE = params.INPUT_SIZE;
        NeuralNet.OUTPUT_SIZE = params.OUTPUT_SIZE;
        NeuralNet.BIAS_START_INDEX = params.BIAS_START_INDEX;
        NeuralNet.INPUT_START_INDEX = params.INPUT_START_INDEX;
        NeuralNet.OUTPUT_START_INDEX = params.OUTPUT_START_INDEX;
        NeuralNet.HIDDEN_START_INDEX = params.HIDDEN_START_INDEX;
        Population.MAXIMUM_STAGNATION = params.MAXIMUM_STAGNATION;
        Population.POPULATION_SIZE = params.POPULATION_SIZE;
        Population.DEFAULT_NETWORK_SIZE = params.DEFAULT_NETWORK_SIZE;
        Chromosome.WEIGHT_MUTATION_CHANCE = params.WEIGHT_MUTATION_CHANCE;
        Chromosome.NODE_MUTATION_CHANCE = params.NODE_MUTATION_CHANCE;
        Chromosome.LINK_MUTATION_CHANCE = params.LINK_MUTATION_CHANCE;
        Chromosome.ENABLE_MUTATION_CHANCE = params.ENABLE_MUTATION_CHANCE;
        Chromosome.DISABLE_MUTATION_CHANCE = params.DISABLE_MUTATION_CHANCE;
        Chromosome.DISJOINT_COEFFICIENT = params.DISJOINT_COEFFICIENT;
        Chromosome.EXCESS_COEFFICIENT = params.EXCESS_COEFFICIENT;
        Chromosome.WEIGHT_DIFFERENCE_COEFFICIENT = params.WEIGHT_DIFFERENCE_COEFFICIENT;
        Species.CROSSOVER_CHANCE = params.CROSSOVER_CHANCE;
        Species.COMPATIBILITY_THRESHOLD = params.COMPATIBILITY_THRESHOLD;
        Species.SURVIVAL_THRESHOLD = params.SURVIVAL_THRESHOLD;
        Gene.WEIGHT_MUTATION_RANGE = params.WEIGHT_MUTATION_RANGE;
        this.pop = new Population(this::createChromosomeBlueprint, this::evaluateChromosomeFitness);
    }

    /**
     * Runs the experiment until fitness limit or generation limit is reached
     */
    public void run() {
        LOGGER.info(String.format("Running the experiment"));
        while (pop.getFittestChromosome().getFitness() < params.FITNESS_LIMIT &&
                pop.getGeneration() < params.GENERATION_LIMIT) {
            pop.advance();
        }
    }

    public Chromosome getFittest() {
        return pop.getFittestChromosome();
    }

    /**
     * Returns a blueprint in the form of integer arrays,
     * each containing 2 integers, outgoing neuron id and incoming neuron id
     * in that order.
     * @return A blueprint to create a chromosome with
     */
    abstract public List<Integer[]> createChromosomeBlueprint();
    abstract public double evaluateChromosomeFitness(Chromosome chromosome);

    class Parameters {
        public int INPUT_SIZE = State.ROWS*State.COLS+State.N_PIECES;
        public int OUTPUT_SIZE = 4+State.COLS;
        public int DEFAULT_HIDDEN_SIZE = 0;
        public int DEFAULT_NETWORK_SIZE = 1 + INPUT_SIZE + OUTPUT_SIZE + DEFAULT_HIDDEN_SIZE;
        public int BIAS_START_INDEX = 0;
        public int INPUT_START_INDEX = 1;
        public int OUTPUT_START_INDEX = INPUT_START_INDEX + INPUT_SIZE;
        public int HIDDEN_START_INDEX = OUTPUT_START_INDEX + OUTPUT_SIZE;
        public int GENERATION_LIMIT = 20; //Number of iterations
        public double FITNESS_LIMIT = 1000; //Value for which we automatically end the search

        public int FITNESS_EVALUATIONS = 20; // Number of evaluations performed per chromosome to be averaged
        public int POPULATION_SIZE = 100; // Population Size
        public double SURVIVAL_THRESHOLD = 0.2; // Percentage of species allowed to survive and breed
        public int MAXIMUM_STAGNATION = 20; // Generations of non-improvement before species is culled
        public double WEIGHT_MUTATION_RANGE = 2.5; // Range at which the weight can be increased or decreased by
        public double WEIGHT_MUTATION_CHANCE = 0.25; // Chance of weight of gene being changed
        public double NODE_MUTATION_CHANCE = 0.30; // Chance of inserting a new node
        public double LINK_MUTATION_CHANCE = 0.25; // Chance of inserting a new link
        public double DISABLE_MUTATION_CHANCE = 0.04; // Chance of a gene being disabled
        public double ENABLE_MUTATION_CHANCE = 0.02; // Chance of a gene being enabled
        public double CROSSOVER_CHANCE = 0.05; // Chance of interspecies breeding
        public double COMPATIBILITY_THRESHOLD = 10; // Threshold for measuring species compatibility
        public double DISJOINT_COEFFICIENT = 1; //  Importance of disjoint genes in measuring compatibility
        public double EXCESS_COEFFICIENT = 1; // Coefficient for excess genes
        public double WEIGHT_DIFFERENCE_COEFFICIENT = 3; // Coefficient for average weight difference

        public Parameters(int inputSize, int outputSize, int hiddenSize) {
            this.INPUT_SIZE = inputSize;
            this.OUTPUT_SIZE = outputSize;
            this.DEFAULT_HIDDEN_SIZE = hiddenSize;
            this.DEFAULT_NETWORK_SIZE = 1 + inputSize + outputSize + hiddenSize;
            this.OUTPUT_START_INDEX = INPUT_START_INDEX + INPUT_SIZE;
            this.HIDDEN_START_INDEX = OUTPUT_START_INDEX + OUTPUT_SIZE;
        }
    }
}

class XORExperiment extends Experiment {
    private static final Logger LOGGER = Logger.getLogger( XORExperiment.class.getName() );
    public XORExperiment() {
    	LOGGER.info(String.format("Initializing an XOR Experiment"));
        this.params = new Parameters(2, 1, 0);
        this.params.FITNESS_LIMIT = 0.99999;
        this.params.GENERATION_LIMIT = 300;
        super.setup();
    }

    @Override
    public List<Integer[]> createChromosomeBlueprint() {
        // Connect bias and input nodes to output nodes
        List<Integer[]> chromosomeBlueprint = new ArrayList<>();
        for (int o=params.BIAS_START_INDEX; o<params.OUTPUT_START_INDEX; o++) {
            for (int i=params.OUTPUT_START_INDEX; i<params.HIDDEN_START_INDEX; i++) {
                chromosomeBlueprint.add(new Integer[]{o, i});
            }
        }

        return chromosomeBlueprint;
    }

    @Override
    public double evaluateChromosomeFitness(Chromosome chromosome) {
        Double[][][] concepts = {
                {{0.0, 0.0}, {0.0}},
                {{0.0, 1.0}, {1.0}},
                {{1.0, 0.0}, {1.0}},
                {{1.0, 1.0}, {0.0}}
        };
        double error = 0;

        NeuralNet nn = new NeuralNet(chromosome);
        Double[] outputs;
        for (Double[][] c: concepts) {
            outputs = nn.activate(Arrays.asList(c[0])).toArray(new Double[1]);
            error += Math.pow(c[1][0] - outputs[0], 2);
        }

        return 1.0 - error;
    }
}

class TetrisExperiment extends Experiment {
    private static final Logger LOGGER = Logger.getLogger( TetrisExperiment.class.getName() );
    private static final int inputSize = State.ROWS * State.COLS + State.N_PIECES;
    private static final int outputSize = 4 * State.COLS;
    private static final int hiddenSize = 1;

    public TetrisExperiment() {
        this.params = new Parameters(inputSize, outputSize, hiddenSize);
        this.params.FITNESS_LIMIT = 1000;
        this.params.GENERATION_LIMIT = 100;
        this.params.POPULATION_SIZE = 100;
        this.params.FITNESS_EVALUATIONS = 10;
        this.params.COMPATIBILITY_THRESHOLD = 10;
        super.setup();
    }

    @Override
    public List<Integer[]> createChromosomeBlueprint() {
        List<Integer[]> chromosomeBlueprint = new ArrayList<>();
        // Connect bias and input nodes to 5 hidden nodes
        for (int o=params.BIAS_START_INDEX; o<params.OUTPUT_START_INDEX; o++) {
            for (int i = params.HIDDEN_START_INDEX; i<params.DEFAULT_NETWORK_SIZE; i++) {
                chromosomeBlueprint.add(new Integer[]{o, i});
            }
        }
        // Connect 5 hidden nodes to output nodes
        for (int o = params.HIDDEN_START_INDEX; o<params.DEFAULT_NETWORK_SIZE; o++) {
            for (int i=params.OUTPUT_START_INDEX; i<params.HIDDEN_START_INDEX; i++) {
                chromosomeBlueprint.add(new Integer[]{o, i});
            }
        }
        return chromosomeBlueprint;
    }

    @Override
    public double evaluateChromosomeFitness(Chromosome chromosome) {
        return (new EvaluateForTetrisTask(chromosome)).compute();
    }


    /**
     * Evaluates the fitness task for that chromosome a number of times equal to Params.FITNESS_EVALUATIONS,
     * then returns the average of those fitness evaluations.
     *
     * Use by passing in the chromosome, then call create subtasks which will spawn the required
     * number of worker threads to do the simulation.
     */
    class EvaluateForTetrisTask extends RecursiveTask<Double> {
        private Chromosome chromosome;
        private boolean isSubTask;

        public EvaluateForTetrisTask(Chromosome chromosome) {
            this.chromosome = chromosome;
            this.isSubTask = false;
        }
        public EvaluateForTetrisTask(Chromosome chromosome, boolean isSubTask) {
            this.chromosome = chromosome;
            this.isSubTask = isSubTask;
        }
        @Override
        protected Double compute() {
            if (!isSubTask) {
                return ForkJoinTask.invokeAll(createSubtasks())
                        .stream()
                        .mapToDouble(ForkJoinTask::join)
                        .sum() / params.FITNESS_EVALUATIONS;
            } else {
    			return evaluateChromosomeFitness();
            }
        }

        private Collection<EvaluateForTetrisTask> createSubtasks() {
            List<EvaluateForTetrisTask> dividedTasks = new ArrayList<>();
            for (int i=0; i<params.FITNESS_EVALUATIONS; i++)
                dividedTasks.add(new EvaluateForTetrisTask(chromosome, true));
            return dividedTasks;
        }

        private double evaluateChromosomeFitness() {
            State s = new State();
            NeuralNet nn = new NeuralNet(chromosome);
            int moves = 0;

            while(!s.hasLost()) {
                nn.activate(StateUtils.normalize(s));
                List<Double> output = nn.getOutput();

                int orient = StateUtils.getOrient(s, output);
                int slot = StateUtils.getSlot(s, orient, output);
                if (slot == -1) {
                    s.lost = true;
                    continue;
                }
                s.makeMove(orient, slot);
            }

            double fitness = (double) s.getRowsCleared();
            fitness += StateUtils.getPercentageFilled(s);
            return fitness;
        }
    }
}

class Population {
    private static final Logger LOGGER = Logger.getLogger( Population.class.getName() );
    public static int MAXIMUM_STAGNATION;
    public static int POPULATION_SIZE;
    public static int DEFAULT_NETWORK_SIZE;
    private int chromosomeCount;
    private int speciesCount;
    private int generation;
    private Innovator innovator;
    private List<Chromosome> chromosomes;
    private List<Species> species;
    private Function<Chromosome, Double> chromosomeFitnessEvaluator;

    public Population(Supplier<List<Integer[]>> chromosomeBlueprintCreator,
                      Function<Chromosome, Double> chromosomeFitnessEvaluator) {
        LOGGER.info(String.format("Creating a new population"));
        this.chromosomeCount = 0;
        this.speciesCount = 0;
        this.chromosomes = new ArrayList<>(POPULATION_SIZE);
        this.species = new ArrayList<>();
        this.innovator = new Innovator(DEFAULT_NETWORK_SIZE);
        this.chromosomeFitnessEvaluator = chromosomeFitnessEvaluator;
        this.generation = 0;
        this.populate(
                createDefaultChromosome(chromosomeBlueprintCreator.get()));
        this.evaluateFitness();
        this.allocateChromosomesToSpecies();
		this.allocateOffsprings();
    }

    /**
     * Populate the population by copying the base chromosome and mutating them
     * @param base Base chromosome to make copies of
     */
    private void populate(Chromosome base) {
        for (int i=0; i<POPULATION_SIZE; i++)
            chromosomes.add((new Chromosome(base)).mutate());
    }

    /**
     * Create a base chromosome from the supplied blueprint
     * @param blueprint The blueprint to create the base chromosome with
     * @return The base chromosome created from the blueprint
     */
    private Chromosome createDefaultChromosome(List<Integer[]> blueprint) {
        LOGGER.fine(String.format("Creating a new default chromosome"));
        List<Gene> genes = new ArrayList<>();
        for (Integer[] b: blueprint)
            genes.add(this.innovator.innovateLink(b[0], b[1]));
        return new Chromosome(this, genes, DEFAULT_NETWORK_SIZE);
    }

    /**
     * @return A new unique chromosome ID
     */
    public int getNewChromosomeId() {
        return chromosomeCount++;
    }

    /**
     * @return A new unique species ID
     */
    public int getNewSpeciesId() {
        return speciesCount++;
    }

    /**
     * Advance the population to the next generation
     * Precondition: All chromosomes must have been evaluated for fitness and separated into species,
	 * and offsprings allocated
     * Postcondition: A new generation with all chromosomes evaluated for fitness and separated,
	 * and offsprings allocated
     * into species
     */
    public void advance() {
        generation++;
        innovator.clearInnovations();
        chromosomes.clear();
        LOGGER.info(String.format("Entering Generation: %d", generation));
        LOGGER.fine(String.format("Slaughtering the weak of each species"));
        for (Species s: species)
            s.cull();
        LOGGER.fine(String.format("Creating the next generation"));
        for (Species s: species) {
            chromosomes.addAll(s.produceAllocatedOffsprings());
            s.clear();
        }
        evaluateFitness();
        allocateChromosomesToSpecies();
        LOGGER.fine(String.format("Pruning extinct species"));
        species.removeIf(s -> s.size() <= 0);
        LOGGER.fine(String.format("Allocate offsprings to species"));
        allocateOffsprings();
    }

    /**
     * Evaluate fitness of population
     */
    private void evaluateFitness() {
        LOGGER.fine(String.format("Evaluating population fitness"));
        ForkJoinPool.commonPool().invoke(
                new EvaluatePopulationFitnessTask(chromosomes, chromosomeFitnessEvaluator));
        LOGGER.info(String.format("Population max fitness: %f", Collections.max(chromosomes).getFitness()));
    }

    /**
     * Allocates chromosomes to existing species
     */
    private void allocateChromosomesToSpecies() {
        LOGGER.fine(String.format("Allocating chromosomes into species"));
        for (Chromosome c: chromosomes) {
            found: {
                for (Species s: species) {
                    if (s.compatibleWith(c)) {
                        LOGGER.finest(String.format("Adding C%d to S%d",
                                c.getId(),
                                s.getId()));
                        s.add(c);
                        break found;
                    }
                }
                species.add(new Species(this, c));
                LOGGER.finer(String.format("Species not found, creating S%d with C%d as rep",
                        species.get(species.size()-1).getId(),
                        c.getId()));
            }
        }
        // Species final computation
        LOGGER.fine(String.format("All species allocated, calcuating species fitness"));
        for (Species s: species)
            s.confirmSpecies();
    }

    /**
     * Allocate no. of offsprings allowed to each species
     */
    private void allocateOffsprings() {
        double averageSum = species.stream()
                .mapToDouble(Species::getAverageFitness)
                .sum();
        for (Species s: species)
            s.setAllocatedOffsprings((int)(s.getAverageFitness()/averageSum*POPULATION_SIZE));
    }

    /**
     * @return The fittest chromosome
     */
    public Chromosome getFittestChromosome() {
        return Collections.max(chromosomes);
    }

    /**
     * @return Current generation number
     */
    public int getGeneration() {
        return this.generation;
    }

    /**
     * Used by species to retrieve a random chromosome from another species for crossover
     * @return A random chromosome from a random species
     */
    public Chromosome getRandomChromosomeFromSpecies() {
        Chromosome rand = null;
        while (rand == null) {
            rand = species.get((new Random().nextInt(species.size()))).getRandomChromosome();
        }
        return rand;
    }

    public Innovator getInnovator() {
        return innovator;
    }

    /**
     * Task for parallelized evaluations of population fitness
     * Sets fitness in the chromosome on completion
     */
    class EvaluatePopulationFitnessTask extends RecursiveAction {
        private List<Chromosome> population;
        private Chromosome chromosome;
        private Function<Chromosome, Double> chromosomeFitnessEvaluator;
        private boolean isSubTask;

        public EvaluatePopulationFitnessTask(List<Chromosome> population,
                                             Function<Chromosome, Double> chromosomeFitnessEvaluator) {
            this.population = population;
            this.chromosomeFitnessEvaluator = chromosomeFitnessEvaluator;
            this.isSubTask = false;
        }
        public EvaluatePopulationFitnessTask(Chromosome chromosome,
                                             Function<Chromosome, Double> chromosomeFitnessEvaluator,
                                             boolean isSubTask) {
            this.chromosome = chromosome;
            this.chromosomeFitnessEvaluator = chromosomeFitnessEvaluator;
            this.isSubTask = isSubTask;
        }

        @Override
        protected void compute() {
            if (!isSubTask) {
                ForkJoinTask.invokeAll(createSubtasks());
            } else {
                evaluateChromosomeFitness();
            }
        }

        private Collection<EvaluatePopulationFitnessTask> createSubtasks() {
            List<EvaluatePopulationFitnessTask> dividedTasks = new ArrayList<>();
            for (Chromosome c: population) {
                dividedTasks.add(new EvaluatePopulationFitnessTask(c, chromosomeFitnessEvaluator, true));
            }
            return dividedTasks;
        }

        private void evaluateChromosomeFitness() {
            chromosome.setFitness(chromosomeFitnessEvaluator.apply(chromosome));
            LOGGER.finest(String.format("C%d has a fitness of %f",
                    chromosome.getId(),
                    chromosome.getFitness()));
        }
    }

    /**
     * Handles the creation of new innovations (i.e. genes)
     */
    class Innovator {
        private int innovationCount;
        private int neuronCount;
        private Map<Integer[], Gene> linkInnovations;
        private Map<Integer[], Gene[]> nodeInnovations;

        /**
         * @param neuronCount The initial number of neurons in the network
         */
        public Innovator(int neuronCount) {
            this.innovationCount = 0;
            this.neuronCount = neuronCount;
            this.linkInnovations = new HashMap<>();
            this.nodeInnovations = new HashMap<>();
        }

        /**
         * Create a new link innovation if it has yet to exist and clone it
         * @param from Neuron giving output
         * @param to Neuron receiving input
         * @return A new gene connecting the neurons
         */
        public Gene innovateLink(int from, int to) {
            LOGGER.finest(String.format("Link innovation requested %d -> %d",
                    from, to));
            Integer[] key = new Integer[]{from, to};
            if (!linkInnovations.containsKey(key)) {
                LOGGER.finest(String.format("Link innovation %d -> %d did not exist, creating",
                        from, to));
                linkInnovations.put(key, new Gene(getNewInnovationId(), from, to));
            }
            return (new Gene(linkInnovations.get(key))).mutateWeight();
        }

        /**
         * Create a new node innovation if it has yet to exist and clone it
         * @param from Neuron giving output
         * @param to Neuron receiving input
         * @return 2 new genes connecting the neurons to a hidden neuron
         */
        public Gene[] innovateNode(int from, int to, double weight) {
            Integer[] key = new Integer[]{from, to};
            if (!nodeInnovations.containsKey(key)) {
                int hiddenNeuron = getNewNeuronId();
                nodeInnovations.put(key, new Gene[]{
                        new Gene(getNewInnovationId(), from, hiddenNeuron),
                        new Gene(getNewInnovationId(), hiddenNeuron, to),
                });
            }
            Gene[] toCopy = nodeInnovations.get(key);
            Gene[] newGenes = new Gene[2];
            newGenes[0] = (new Gene(toCopy[0]));
            newGenes[0].weight = 1.0;
            newGenes[1] = (new Gene(toCopy[1]));
            newGenes[1].weight = weight;
            return newGenes;
        }

        /**
         * Clear the innovation maps
         */
        public void clearInnovations() {
            linkInnovations.clear();
            nodeInnovations.clear();
        }

        /**
         * @return A new unique innovation ID
         */
        private int getNewInnovationId() {
            return innovationCount++;
        }

        /**
         * @return A new hidden neuron ID
         */
        private int getNewNeuronId() {
            return neuronCount++;
        }
    }
}
