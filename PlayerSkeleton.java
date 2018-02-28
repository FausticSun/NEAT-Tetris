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

			// populate chromosomes
			List<Chromosone> population = new ArrayList<Chromosone>();
			for (int i=0; i<Params.POPULATION_SIZE; i++){
				population.add(Chromosone.createDefaultChromosone());
			}

			// mutate population
			for (Chromosone chromosone : population)
				chromosone.mutate();

			List<Species> speciesList = new ArrayList<Species>();
			for (Chromosone chromosone : population)
				findSpecies(chromosone, speciesList);

			//run NEAT
			Chromosone fittestChromosone = population.get(0);
			List<Chromosone> newChildren;
			double highestFitness = -1;
			int currentStagnation = 0;
			double totalSpeciesFitness;
			for (int i = 0; i < Params.GENERATION_LIMIT; i++) {
				System.out.println("Current generation: " + i);
				//evaluate fitness of chromosones
				//double fitness;
				forkJoinPool.invoke(
						new EvaluatePopulationFitnessTask(population));
				//fitness = forkJoinPool.invoke(
				//	new EvaluateChromosoneFitnessTask(Chromosone.createDefaultChromosone()));
				for (Chromosone chromosone : population)
					System.out.println("Chromosone #: " + chromosone.id + ", fitness: " + chromosone.fitness);

				// TODO if fitness limit reached, break

				// TODO cull stagnant species if max not improving

				// TODO if survival threshhold not reached, cull underperformers in each species

				// TODO breed children in each species and mutate them
				// evaluate which species children get put into
				// note: children can be crossbred
			}
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

	/**
	 * checks the current chromosone and adds it to a species list
	 * @param chromosone - chromosone to be added to a species list
	 */
	public void findSpecies(Chromosone chromosone, List<Species> speciesList) {
		for (Species species : speciesList) {
			if (chromosone.computeGeneDistance(species.representative) < Params.COMPATIBILITY_THRESHOLD) {
				species.speciesPopulation.add(chromosone);
				//System.out.println("chromosome added to species");
				return;
			}
		}
		//did not fit into any species, creating new species
		speciesList.add(new Species(chromosone));
		//System.out.println("chromosome created new species");
	}

	/**
	 * updates current best chromosone to the one with best fitness
	 * @param population - list of all current chromosones with updated fitness
	 * @param fittestChromosone - current best chromosone
	 */
	public void updateFittest(List<Chromosone> population, Chromosone fittestChromosone){
		for (Chromosone chromosone : population) {
			if (fittestChromosone.fitness < chromosone.fitness)
				fittestChromosone = chromosone;
		}
	}

	//implement this function to have a working system
	public int pickMove(State s, int[][] legalMoves) {
		return 0;
	}
}

class Species {
	public Chromosone representative;
	public List<Chromosone> speciesPopulation;
	public double averageFitness;
	public int speciesID;

	public Species(Chromosone representative){
		speciesPopulation = new ArrayList<Chromosone>();
		this.representative = representative;
		speciesPopulation.add(representative);
		speciesID = Globals.getSpeciesId();
	}

	/**
	 * aggressively removes weaklings from the population
	 */
	public void cull(List<Chromosone> population) {
		Collections.sort(speciesPopulation);
		int limit = (int)Math.ceil(speciesPopulation.size() * Params.SURVIVAL_THRESHOLD);
		speciesPopulation = speciesPopulation.subList(0, limit);
		for (int i=limit; i< speciesPopulation.size(); i++) {
			population.remove(speciesPopulation.get(i));
		}
	}

	/**
	 *
	 * @return returns the average fitness of the chromosones in the species
	 */
	public double computeAverageFitness() {
		if (speciesPopulation.size() == 0)
			return 0;
		double totalFitness = 0;
		for (Chromosone chromosone : speciesPopulation)
			totalFitness += chromosone.fitness;
		System.out.println("Species ID: " + speciesID + ", PopSize : " + speciesPopulation.size() + ", AverageFitness = " + totalFitness/speciesPopulation.size());
		averageFitness = totalFitness/speciesPopulation.size();
		return averageFitness;
	}

	/**
	 * creates baby chromosones equal to the amount requested
	 * if there is only 1 parent, it will always crossbreed
	 * otherwise, it will always breed with a different parent
	 * @param numberOfChildren - number of chromosones to populate return with
	 * @param population - in case of cross-breeding
	 * @return returns a list of new baby chromosones
	 */
	public List<Chromosone> breed(int numberOfChildren, List<Chromosone> population) {
		List<Chromosone> newChildren = new ArrayList<Chromosone>();
		System.out.println("Species ID: " + speciesID + " breeding " + numberOfChildren + " new children.");
		Chromosone parent1, parent2;
		for (int i=0; i<numberOfChildren; i++) {
			parent1 = speciesPopulation.get((int)Math.floor(Math.random() * speciesPopulation.size()));
			parent2 = parent1;
			if (Math.random() < Params.CROSSOVER_CHANCE || speciesPopulation.size() == 1) {//crossbreed with anything in pop
				while (parent1 == parent2)
					parent2 = population.get((int)Math.floor(Math.random() * population.size()));
			}
			else { //not crossbreed
				while (parent1 == parent2)
					parent2 = speciesPopulation.get((int)Math.floor(Math.random() * speciesPopulation.size()));
			}
			newChildren.add(parent1.breedWith(parent2));
		}
		return newChildren;
	}
}
// Feed-forward neural network
// Neurons are arranged in the List from Input, Output and Hidden
class NeuralNet {
	public List<Neuron> neurons;
	public Chromosone chromosone;
	
	/**
	 * A neural net built based on a specific chromosone data.
	 * Call activate() to pass in the input and then getOuput() later.
	 *
	 * @param chromosone The chromosone to create the neural net information.
	 */
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
			if(g.isEnabled) {
				n = neurons.get(g.to);
				n.incomingNeurons.add(neurons.get(g.from));
				n.incomingWeights.add(g.weight);
			}
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
	
	/**
	 * Places these inputs into the neural net input neurons.
	 *
	 * @param inputs The list of inputs from the screen and the current piece.
	 * @return True if successful, false otherwise.
	 */
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
	
	/**
	 * Gets the output of the neural net.
	 * Call this after you call activate and pass in the inputs.
	 *
	 * @return The list of outputs from the chromosone.
	 */
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

/**
 * Neuron refers to a specific node, be in input, output, hidden or bias. 
 * This stores information about the node, incoming neurons and their weights.
 */
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
	
	/**
	 * Recursively activate dependent neurons
	 */
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

class Gene implements Comparable<Gene>{
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

	public void mutate() {
		if (Math.random() < Params.DISABLE_MUTATION_CHANCE)
			mutateDisable();
		if (Math.random() < Params.ENABLE_MUTATION_CHANCE)
			mutateEnable();
		if (Math.random() < Params.WEIGHT_MUTATION_CHANCE)
			mutateWeight();
	}

	public int compareTo(Gene other) {
		return id - other.id;
	}

	/**
	 * mutates by disabling a link
	 */
	public void mutateDisable() {
		isEnabled = false;
	}

	/**
	 * mutates by enabling a link
	 */
	public void mutateEnable() {
		isEnabled = true;
	}

	/**
	 * mutates by changing weight
	 */
	public void mutateWeight() {
		if (Math.random() < 0.5)
			weight += Math.random() * Params.WEIGHT_MUTATION_RANGE;
		else
			weight -= Math.random() * Params.WEIGHT_MUTATION_RANGE;
	}
}

class Chromosone implements Comparable<Chromosone>{
	public int neuronCount;
	public List<Gene> genes;
	public double fitness;
	public int id;

	public Chromosone() {
		neuronCount = Globals.NEURON_COUNT;
		genes = new ArrayList<Gene>();
		fitness = -1;
		id = Globals.getChromosoneId();
	}

	public Chromosone(Chromosone other) {
		this.neuronCount = other.neuronCount;
		this.genes = new ArrayList<Gene>();
		for (Gene g: other.genes)
			this.genes.add(new Gene(g));
		this.fitness = other.fitness;
		this.id = Globals.getChromosoneId();
	/**
	 *
	 * @param other - other chromosone being tested against
	 * @return positive if better, negative if worse
	 */
	public int compareTo(Chromosone other) {
		if (this.fitness > other.fitness)
			return 1;
		else if (this.fitness > other.fitness)
			return -1;
		else
			return 0;
	}

	/**
	 * Breeds 2 chromosones together:
	 *  Same ID genes get randomly picked from one parent
	 *  Excess and Disjoint genes get picked from the better parent
	 *  If both parents are the same, Excess and Disjoint genes get added from both
	 * @param other - other chromosone being bred with
	 * @return the baby chromosone
	 */
	public Chromosone breedWith(Chromosone other) {
		Chromosone chromosone = new Chromosone();
		// TODO proper algorithm. I was braindead and did the not efficient one
		if (fitness >= other.fitness) { //this is base chromosone
			boolean matchFound;
			for (Gene gene1 : genes) {
				matchFound = false;
				for (Gene gene2 : other.genes) { //check if it has a match
					if (gene1.id == gene2.id) {
						if (Math.random() < 0.5)
							chromosone.genes.add(gene1);
						else
							chromosone.genes.add(gene2);
						matchFound = true;
						break;
					}
					if (gene1.id < gene2.id) { //too far to have a match
						break;
					}
				}
				//no match
				if (!matchFound) {
					chromosone.genes.add(gene1);
				}
			}
			chromosone.neuronCount = neuronCount;
		}
		if  (other.fitness > fitness) { //other is base chromosone
			boolean matchFound;
			for (Gene gene1 : other.genes) {
				matchFound = false;
				for (Gene gene2 : genes) { //check if it has a match
					if (gene1.id == gene2.id) {
						if (Math.random() < 0.5)
							chromosone.genes.add(gene1);
						else
							chromosone.genes.add(gene2);
						matchFound = true;
						break;
					}
					if (gene1.id < gene2.id) { //too far to have a match
						break;
					}
				}
				//no match
				if (!matchFound) {
					chromosone.genes.add(gene1);
				}
			}
			chromosone.neuronCount = other.neuronCount;
		}
		if (fitness == other.fitness) {//both same fitness
			boolean matchFound;
			//add other's disjoint and excess genes
			for (Gene gene1 : other.genes) {
				matchFound = false;
				for (Gene gene2 : genes) { //check if it has a match
					if (gene1.id == gene2.id) {
						matchFound = true;
						break;
					}
					if (gene1.id < gene2.id) { //too far to have a match
						break;
					}
				}
				//no match
				if (!matchFound) {
					chromosone.genes.add(gene1);
				}
			}
			chromosone.neuronCount = Math.max(neuronCount, other.neuronCount);
		}
		return chromosone;
	}
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

	public Chromosone clone() {
		Chromosone c = new Chromosone();
		c.genes.addAll(this.genes);
		c.fitness = this.fitness;
		//TODO set ID maybe?
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
}

/**
 * Evaluates the fitness task for that chromosone a number of times equal to Params.FITNESS_EVALUATIONS,
 * then returns the average of those fitness evaluations.
 *
 * Use by passing in the chromosone, then call create subtasks which will spawn the required
 * number of worker threads to do the simulation.
 */
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
		System.out.printf("Chromsone #%d fitness computed with thread %s%n", chromosone.id, Thread.currentThread().getName());
		return fitness;
	}
}

class EvaluatePopulationFitnessTask extends RecursiveAction {
	private List<Chromosone> population;
	private Chromosone chromosone;
	private boolean isSubTask;

	public EvaluatePopulationFitnessTask(List<Chromosone> population) {
		this.population = population;
		this.isSubTask = false;
	}
	public EvaluatePopulationFitnessTask(Chromosone chromosone, boolean isSubTask) {
		this.chromosone = chromosone;
		this.isSubTask = isSubTask;
	}

	@Override
	protected void compute() {
		if (!isSubTask) {
			ForkJoinTask.invokeAll(createSubtasks());
		} else {
			evaluateChromosoneFitness();
		}
	}

	private Collection<EvaluatePopulationFitnessTask> createSubtasks() {
		List<EvaluatePopulationFitnessTask> dividedTasks = new ArrayList<EvaluatePopulationFitnessTask>();
		for (Chromosone c: population) {
			dividedTasks.add(new EvaluatePopulationFitnessTask(c, true));
		}
		return dividedTasks;
	}

	private void evaluateChromosoneFitness() {
		chromosone.fitness = (new EvaluateChromosoneFitnessTask(chromosone).compute());
	}
}

class Params {
	public static final int INPUT_SIZE = State.ROWS*State.COLS+State.N_PIECES;
	public static final int OUTPUT_SIZE = 4+State.COLS;
	public static final int BIAS_START_INDEX = 0;
	public static final int INPUT_START_INDEX = 1;
	public static final int OUTPUT_START_INDEX = INPUT_START_INDEX + INPUT_SIZE;
	public static final int HIDDEN_START_INDEX = OUTPUT_START_INDEX + OUTPUT_SIZE;
	public static final int GENERATION_LIMIT = 1; // Number of iterations
	public static final double FITNESS_LIMIT = 1000; // Value for which we automatically end the search
	
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
	public static int CHROMOSONE_COUNT = 0;

	public static int getInnovationId() {
		INNOVATION_COUNT++;
		return INNOVATION_COUNT;
	}

	public static int getChromosoneId() {
		CHROMOSONE_COUNT++;
		return CHROMOSONE_COUNT;
	}
}