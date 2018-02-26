import java.util.*;
import java.util.concurrent.*;
import java.lang.StringBuilder;

public class PlayerSkeleton {
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
		ForkJoinPool forkJoinPool = new ForkJoinPool();
		if (EVOLVE) {
			double fitness;
			fitness = forkJoinPool.invoke(
				new EvaluateChromosoneFitnessTask(Chromosone.createDefaultChromosone()));
			System.out.println(fitness);
			return;
		}

		State s = new State();
		NeuralNet nn = new NeuralNet(Chromosone.createDefaultChromosone());

		if (!HEADLESS)
			new TFrame(s);
		while(!s.hasLost()) {
			nn.activate(StateUtils.normalize(s));
			List<Double> output = nn.getOutput();
			
			// DEUBG: Output output weights
			// for (double d: output)
			// 	System.out.format("%.3f ", d);
			// System.out.format("%n");

			int orient = StateUtils.getOrient(s, output);
			int slot = StateUtils.getSlot(s, orient, output);
			if (slot == -1) {
				s.lost = true;
				continue;
			}

			s.makeMove(orient, slot);

			if (!HEADLESS) {
				s.draw();
				s.drawNext(0,0);
				try {
					Thread.sleep(300);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
			}
		}
		System.out.println("You have completed "+s.getRowsCleared()+" rows.");
	}

	//implement this function to have a working system
	public int pickMove(State s, int[][] legalMoves) {
		return 0;
	}
}

// Feed-forward neural network
// Neurons are arranged in the List from Input, Output and Hidden
class NeuralNet {
	public List<Neuron> neurons;
	public Chromosone chromosone;
	public NeuralNet(Chromosone chromosone) {
		this.chromosone = chromosone;
		
		// DEBUG: Prints the chromosone on creation
		// for (Gene g: chromosone.genes) {
		// 	System.out.printf("ID: %d, From: %d, To: %d%n", g.id, g.from, g.to);
		// }

		// Create Neurons
		neurons = new ArrayList<Neuron>();
		for (int i=0; i<chromosone.neuronCount; i++)
			neurons.add(new Neuron());

		// Insert links
		Neuron n;
		for (Gene g: chromosone.genes) {
			n = neurons.get(g.to);
			n.incomingNeurons.add(neurons.get(g.from));
			n.incomingWeights.add(g.weight);
		}

		// Setup Bias Neuron
		neurons.get(0).type = ActivationType.BIAS;
		neurons.get(0).isActive = true;
		
		// Setup Input Neurons
		for (int i=Params.INPUT_START_INDEX; i<Params.OUTPUT_START_INDEX; i++) {
			n = neurons.get(i);
			n.type = ActivationType.LINEAR;
			n.isActive = true;
		}

		// Setup Output Neurons
		for (int i=Params.OUTPUT_START_INDEX; i<Params.HIDDEN_START_INDEX; i++) {
			neurons.get(i).type = ActivationType.SIGMOID;
		}

		// Setup Hidden Neurons
		for (int i=Params.HIDDEN_START_INDEX; i<neurons.size(); i++) {
			neurons.get(i).type = ActivationType.SIGMOID;
		}
	}

	public boolean activate(List<Double> inputs) {
		// Check input size
		if (inputs.size() != Params.INPUT_SIZE) {
			System.out.println("Input size mismatch!");
			return false;
		}

		// Clear the network
		reset();

		// Set Input Neurons
		for (int i=0; i<Params.INPUT_SIZE; i++) {
			neurons.get(Params.INPUT_START_INDEX+i).value = inputs.get(i);
		}

		// Activate Output Neurons
		for (int i=Params.OUTPUT_START_INDEX; i<Params.HIDDEN_START_INDEX; i++) {
			neurons.get(i).activate();
		}

		return true;
	}

	public List<Double> getOutput() {
		List<Double> output = new ArrayList<Double>();
		for (int i=Params.OUTPUT_START_INDEX; i<Params.HIDDEN_START_INDEX; i++) {
			output.add(neurons.get(i).value);
		}
		return output;
	}

	public void reset() {
		for (int i=Params.OUTPUT_START_INDEX; i<neurons.size(); i++) {
			neurons.get(i).isActive = false;
		}
	}
}

class Neuron {
	public List<Neuron> incomingNeurons;
	public List<Double> incomingWeights;
	public ActivationType type;
	public double value;
	public boolean isActive;

	public Neuron() {
		this.incomingNeurons = new ArrayList<Neuron>();
		this.incomingWeights = new ArrayList<Double>();
		this.type = ActivationType.SIGMOID;
		this.value = 0;
		this.isActive = false;
	}

	// Recursively activate dependent neurons
	public void activate() {
		double sum = 0;
		Neuron n;
		for (int i=0; i<incomingNeurons.size(); i++) {
			n = incomingNeurons.get(i);
			if(!n.isActive) {
				n.activate();
			}
			sum += n.value * incomingWeights.get(i);
		}
		switch (this.type) {
			case BIAS: this.value = 1;
			case LINEAR: this.value = sum; break;
			case SIGMOID: this.value = Calc.sigmoid(sum); break;
		}
		this.isActive = true;
	}
}

enum ActivationType {
	LINEAR, SIGMOID, BIAS
}

class Calc {
	public static double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}
}

class Population {
	List<Chromosone> chromosones;
	List<Species> species;
}

class Species {
	List<Chromosone> chromosones;
	Chromosone representative;
}

class Gene {
	public int id;
	public int from;
	public int to;
	public double weight;
	public boolean isEnabled;

	public Gene() {
		this.id = 0;
		this.from = 0;
		this.to = 0;
		this.weight = 0;
		this.isEnabled = true;
	}

	public Gene(int id, int from, int to, double weight) {
		this.id = id;
		this.from = from;
		this.to = to;
		this.weight = weight;
		this.isEnabled = true;
	}
}

class Chromosone {
	public int neuronCount;
	public List<Gene> genes;

	public Chromosone() {
		neuronCount = Globals.NEURON_COUNT;
		genes = new ArrayList<Gene>();
	}

	public static Chromosone createDefaultChromosone() {
		Chromosone c = new Chromosone();
		int hid = Params.HIDDEN_START_INDEX;
		// Connect input neurons to a single hidden neuron
		for (int i=0; i<Params.OUTPUT_START_INDEX; i++) {
			c.genes.add(new Gene(Globals.getInnovationId(), i, hid, Math.random()*2-1));
		}
		// Connect single hidden neuron to output neurons
		for (int i=Params.OUTPUT_START_INDEX; i<Params.HIDDEN_START_INDEX; i++) {
			c.genes.add(new Gene(Globals.getInnovationId(), hid, i, Math.random()*2-1));
		}

		return c;
	}
}

class FittestChromosone {
	public String xml;
	public FittestChromosone() {
		StringBuilder sb = new StringBuilder();
		sb.append("");
		xml = sb.toString();
	}
}

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

	// Return -1 if dropping in an invalid slot
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
}

class EvaluateChromosoneFitnessTask extends RecursiveTask<Double> {
	private Chromosone chromosone;
	private boolean isSubTask;
	public EvaluateChromosoneFitnessTask(Chromosone chromosone) {
		this.chromosone = chromosone;
		this.isSubTask = false;
	}
	public EvaluateChromosoneFitnessTask(Chromosone chromosone, boolean isSubTask) {
		this.chromosone = chromosone;
		this.isSubTask = isSubTask;
	}
	@Override
	protected Double compute() {
		if (!isSubTask) {
			return ForkJoinTask.invokeAll(createSubtasks())
				.stream()
				.mapToDouble(ForkJoinTask::join)
				.sum() / Params.FITNESS_EVALUATIONS;
		} else {
			return evaluateChromosoneFitness();
		}
	}

	private Collection<EvaluateChromosoneFitnessTask> createSubtasks() {
		List<EvaluateChromosoneFitnessTask> dividedTasks = new ArrayList<>();
		for (int i=0; i<Params.FITNESS_EVALUATIONS; i++)
			dividedTasks.add(new EvaluateChromosoneFitnessTask(chromosone, true));
		return dividedTasks;
	}

	private double evaluateChromosoneFitness() {
		State s = new State();
		NeuralNet nn = new NeuralNet(chromosone);
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
			moves += 1;
		}

		double fitness = (double) s.getRowsCleared();
		fitness = fitness == 0 ? moves / 100.0 : fitness;
		return fitness;
	}
}

class Params {
	public static final int INPUT_SIZE = State.ROWS*State.COLS+State.N_PIECES;
	public static final int OUTPUT_SIZE = 4+State.COLS;
	public static final int BIAS_START_INDEX = 0;
	public static final int INPUT_START_INDEX = 1;
	public static final int OUTPUT_START_INDEX = INPUT_START_INDEX + INPUT_SIZE;
	public static final int HIDDEN_START_INDEX = OUTPUT_START_INDEX + OUTPUT_SIZE;
	
	public static final int FITNESS_EVALUATIONS = 20; // Number of evaluations performed per chromosone to be averaged
	public static final int POPULATION_SIZE = 200; // Population Size
	public static final double SURVIVAL_THRESHOLD = 0.2; // Percentage of species allowed to survive and breed
	public static final double MAXIMUM_STAGNATION = 15; // Generations of non-improvement before species is culled
	public static final double WEIGHT_MUTATION_RANGE = 2.5; // Range at which the weight can be increased or decreased by
	public static final double WEIGHT_MUTATION_CHANCE = 0.025; // Chance of weight of gene being changed
	public static final double NODE_MUTATION_CHANCE = 0.03; // Chance of inserting a new node 
	public static final double LINK_MUTATION_CHANCE = 0.05; // Chance of inserting a new link
	public static final double DISABLE_MUTATION_CHANCE = 0.04; // Chance of a gene being disabled
	public static final double ENABLE_MUTATION_CHANCE = 0.02; // Chance of a gene being enabled
	public static final double CROSSOVER_CHANCE = 0.05; // Chance of interspecies breeding
	public static final double COMPATIBILITY_THRESHOLD = 3; // Threshold for measuring species compatibility
	public static final double C1 = 1; // Coefficient for importance of excess genes in measuring compatibility
	public static final double C2 = 1; // Coefficient for disjoint genes
	public static final double C3 = 3; // Coefficient for average weight difference
}

class Globals {
	public static int NEURON_COUNT = Params.HIDDEN_START_INDEX+1;
	public static int INNOVATION_COUNT = 0;

	public static int getInnovationId() {
		INNOVATION_COUNT++;
		return INNOVATION_COUNT;
	}
}