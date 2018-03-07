import java.util.*;
import java.util.concurrent.*;
import java.util.function.*;
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
			List<Chromosome> population = new ArrayList<Chromosome>();
			for (int i=0; i<Params.POPULATION_SIZE; i++){
				population.add(Chromosome.createDefaultChromosome());
			}

			// mutate population
			for (Chromosome chromosome : population)
				chromosome.mutate();

			List<Species> speciesList = new ArrayList<Species>();
			for (Chromosome chromosome : population)
				findSpecies(chromosome, speciesList);

			//run NEAT
			Chromosome fittestChromosome = population.get(0);
			List<Chromosome> newChildren;
			double highestFitness = -1;
			int currentStagnation = 0;
			double totalSpeciesFitness;
			for (int i = 0; i < Params.GENERATION_LIMIT; i++) {
				System.out.println("Current generation: " + i);
				//evaluate fitness of chromosomes
				//double fitness;
				forkJoinPool.invoke(
						new EvaluatePopulationFitnessTask(population));
				//fitness = forkJoinPool.invoke(
				//	new EvaluateChromosomeFitnessTask(Chromosome.createDefaultChromosome()));
				for (Chromosome chromosome : population)
					System.out.println("Chromosome #: " + chromosome.id + ", fitness: " + chromosome.fitness);

				//update fitness and check stagnation
				updateFittest(population, fittestChromosome);
				if (highestFitness < fittestChromosome.fitness) {
					highestFitness = fittestChromosome.fitness;
					currentStagnation = 0;
				} else {
					currentStagnation++;
				}
				//if fitness limit reached, break
				if (fittestChromosome.fitness > Params.FITNESS_LIMIT)
					break;


				// breed new generation
				newChildren = new ArrayList<Chromosome>();
				totalSpeciesFitness = 0;
				for (Species species : speciesList) {
					totalSpeciesFitness += species.computeAverageFitness();
				}
				// check for stagnation
				System.out.println("Current Stagnation: " + currentStagnation);
				if (currentStagnation >= Params.MAXIMUM_STAGNATION || totalSpeciesFitness == 0) { //wipe out everyone by ignoring them
					System.out.println("stagnating population. culling");
					currentStagnation = 0;
					Collections.sort(population);
					Chromosome chromosome;
					for (int j=0; j<Params.POPULATION_SIZE; j++) {
						chromosome = population.get(0).breedWith(population.get(1)); //breed only the fittest 2 chromosomes
						newChildren.add(chromosome);
					}
				}
				else { //do not wipe out everyone

					// cull underperformers in each species
					for (Species species : speciesList)
						species.cull(population);

					//breed in all species
					int numberOfChildren;
					List<Chromosome> speciesChildren;
					for (Species species : speciesList) {
						numberOfChildren = (int)Math.ceil(Params.POPULATION_SIZE*species.averageFitness/totalSpeciesFitness);
						speciesChildren = species.breed(numberOfChildren, population);
						newChildren.addAll(speciesChildren);
					}
				}

				// mutate new children and sort them into species
				speciesList = new ArrayList<Species>();
				for (Chromosome chromosome : newChildren) {
					chromosome.mutate();
					findSpecies(chromosome, speciesList);
				}

				population = new ArrayList<Chromosome>();
				population.addAll(newChildren);
			}

			forkJoinPool.invoke(
					new EvaluatePopulationFitnessTask(population));

			Collections.sort(population);
			FittestChromosome fc = new FittestChromosome(population.get(0));
			// TODO do something with fittest chromosome
			System.out.println("Best fitness: " + population.get(0).fitness);
			return;
		}

		State s = new State();
		NeuralNet nn = new NeuralNet(Chromosome.createDefaultChromosome());

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
	 * checks the current chromosome and adds it to a species list
	 * @param chromosome - chromosome to be added to a species list
	 */
	public void findSpecies(Chromosome chromosome, List<Species> speciesList) {
		for (Species species : speciesList) {
			if (chromosome.computeGeneDistance(species.representative) < Params.COMPATIBILITY_THRESHOLD) {
				species.speciesPopulation.add(chromosome);
				//System.out.println("chromosome added to species");
				return;
			}
		}
		//did not fit into any species, creating new species
		speciesList.add(new Species(chromosome));
		//System.out.println("chromosome created new species");
	}

	/**
	 * updates current best chromosome to the one with best fitness
	 * @param population - list of all current chromosomes with updated fitness
	 * @param fittestChromosome - current best chromosome
	 */
	public void updateFittest(List<Chromosome> population, Chromosome fittestChromosome){
		for (Chromosome chromosome : population) {
			if (fittestChromosome.fitness < chromosome.fitness)
				fittestChromosome = chromosome;
		}
	}

	//implement this function to have a working system
	public int pickMove(State s, int[][] legalMoves) {
		return 0;
	}
}

class Species {
	public Chromosome representative;
	public List<Chromosome> speciesPopulation;
	public double averageFitness;
	public int speciesID;

	public Species(Chromosome representative){
		speciesPopulation = new ArrayList<Chromosome>();
		this.representative = representative;
		speciesPopulation.add(representative);
		speciesID = Globals.getSpeciesId();
	}

	/**
	 * aggressively removes weaklings from the population
	 */
	public void cull(List<Chromosome> population) {
		Collections.sort(speciesPopulation);
		int limit = (int)Math.ceil(speciesPopulation.size() * Params.SURVIVAL_THRESHOLD);
		speciesPopulation = speciesPopulation.subList(0, limit);
		for (int i=limit; i< speciesPopulation.size(); i++) {
			population.remove(speciesPopulation.get(i));
		}
	}

	/**
	 *
	 * @return returns the average fitness of the chromosomes in the species
	 */
	public double computeAverageFitness() {
		if (speciesPopulation.size() == 0)
			return 0;
		double totalFitness = 0;
		for (Chromosome chromosome : speciesPopulation)
			totalFitness += chromosome.fitness;
		System.out.println("Species ID: " + speciesID + ", PopSize : " + speciesPopulation.size() + ", AverageFitness = " + totalFitness/speciesPopulation.size());
		averageFitness = totalFitness/speciesPopulation.size();
		return averageFitness;
	}

	/**
	 * creates baby chromosomes equal to the amount requested
	 * if there is only 1 parent, it will always crossbreed
	 * otherwise, it will always breed with a different parent
	 * @param numberOfChildren - number of chromosomes to populate return with
	 * @param population - in case of cross-breeding
	 * @return returns a list of new baby chromosomes
	 */
	public List<Chromosome> breed(int numberOfChildren, List<Chromosome> population) {
		List<Chromosome> newChildren = new ArrayList<Chromosome>();
		System.out.println("Species ID: " + speciesID + " breeding " + numberOfChildren + " new children.");
		Chromosome parent1, parent2;
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
	public Chromosome chromosome;
	
	/**
	 * A neural net built based on a specific chromosome data.
	 * Call activate() to pass in the input and then getOuput() later.
	 *
	 * @param chromosome The chromosome to create the neural net information.
	 */
	public NeuralNet(Chromosome chromosome) {
		this.chromosome = chromosome;

		// DEBUG: Prints the chromosome on creation
		// for (Gene g: chromosome.genes) {
		// 	System.out.printf("ID: %d, From: %d, To: %d%n", g.id, g.from, g.to);
		// }

		// Create Neurons
		neurons = new ArrayList<Neuron>();

		for (int i=0; i<chromosome.neuronCount+1; i++)
			neurons.add(new Neuron());

		// Insert links
		Neuron n;
		for (Gene g: chromosome.genes) {
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
	 * @return The list of outputs, null if unsuccessful
	 */
	public List<Double> activate(List<Double> inputs) {
		// Check input size
		if (inputs.size() != Params.INPUT_SIZE) {
			System.out.println("Input size mismatch!");
			return null;
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

		// Return output as list
        List<Double> output = new ArrayList<Double>();
        for (int i=Params.OUTPUT_START_INDEX; i<Params.HIDDEN_START_INDEX; i++) {
            output.add(neurons.get(i).value);
        }
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

class Gene implements Comparable<Gene>{
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

    /**
     * Clones a gene from a reference gene and randomize its weight
     * @param other
     * @param isEnabled
     */
	public Gene(Gene other, boolean isEnabled) {
        this.id = other.id;
        this.from = other.from;
        this.to = other.to;
        this.weight = Math.random()*2-1;
        this.isEnabled = isEnabled;
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

class Chromosome implements Comparable<Chromosome> {
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
		System.out.println("Breeding chromosome " + this.id + " with " + other.id);
		//Ensure that this has a higher fitness than other.
		if(other.fitness > this.fitness) {
			return other.breedWith(this);
		}
		Chromosome chromosome = new Chromosome();

		Collections.sort(this.genes);
		Collections.sort(other.genes);
		System.out.print("Printing i: [");
		for(int i = 0; i < this.genes.size() && i < 100; i++) {
			System.out.print(this.genes.get(i).id + " ");
		}
		System.out.println("]");
		System.out.print("Printing j: [");
		for(int j = 0; j < other.genes.size() && j < 100; j++) {
			System.out.print(other.genes.get(j).id + " ");
		}
		System.out.println("]");
		int i = 0, j = 0;
		while(i < genes.size() && j < other.genes.size()) {
			System.out.println("Matching " + i + " with " + j);
			if(genes.get(i).id == other.genes.get(j).id) {
				if (Math.random() < 0.5)
					chromosome.genes.add(this.genes.get(i));
				else
					chromosome.genes.add(other.genes.get(j));
				i++;
				j++;
				System.out.println("They match!");
				continue;
			}
			if(genes.get(i).id > other.genes.get(j).id) {
				if(this.fitness == other.fitness) {
					chromosome.genes.add(other.genes.get(j));
				}
				j++;
				System.out.println("j is lower.");
				continue;
			}
			if(other.genes.get(j).id > genes.get(i).id) {
				chromosome.genes.add(genes.get(i));
				System.out.println("i is lower.");
				i++;
				continue;
			}
		}
		if(i < genes.size() - 1) { //add the rest of the fitter chromosome
			for(; i < genes.size(); i++) {
				chromosome.genes.add(genes.get(i));
			}
		} else {
			//only add if other is as fit as I
			if(other.fitness == this.fitness) {
				for(; j < genes.size(); j++) {
					chromosome.genes.add(other.genes.get(j));
				}
			}
		}
		chromosome.neuronCount = neuronCount;
		if(this.fitness == other.fitness) {
			chromosome.neuronCount = Math.max(this.neuronCount, other.neuronCount);
		}

		return chromosome;
	}

	/**
	 * mutates the chromosome randomly:
	 * each gene can mutate weight or get disabled/enabled
	 * chromosome can gain a new node or link
	 */
	public Chromosome mutate() {
		for (Gene gene : genes)
			gene.mutate();
		if (Math.random() < Params.LINK_MUTATION_CHANCE)
			mutateLink();
		if (Math.random() < Params.NODE_MUTATION_CHANCE)
			mutateNode();
		return this;
	}

	/**
	 * mutates by creating a link
	 * To make it easy for validation for now, only options are
	 * 1) must start from a input node
	 * 2) must end at a output node
	 * TODO: refine with bellman's ford instead
	 */
	public void mutateLink() {
		int startNode = -1;
		int endNode = -1;
		boolean foundPair = false;
		for (int i = 0; i < 10; i++) { //try this 10 times or until success
			startNode = genes.get((int) Math.floor(Math.random() * genes.size())).from;
			endNode = genes.get((int) Math.floor(Math.random() * genes.size())).to;
			foundPair = true;

			if (startNode == endNode)
				foundPair = false;

			//check if the gene is already in the chromosome
			for (Gene gene : genes)
				if (startNode == gene.from && endNode == gene.to)
					foundPair = false;

			//if we found a pair to match, exit loop
			if (foundPair)
				break;
		}

		if (!foundPair) //break if can't find suitable pair
			return;

		System.out.println("Mutating new link between node " + startNode + " and node " + endNode);
		Integer linkID = Globals.INNOVATION_MAP.get(startNode).get(endNode);
		if (linkID == null) { //link does not exist yet
			//check if link fits our easy restriction
			//Probable optimization: Perform DFS from end node to start node to check for links
			//Problems: List of edges (Genes) is stored in a list and there is no way to know the list of
			//edges from 1 node to another other than going through the list.
			if ((startNode < Params.OUTPUT_START_INDEX) && (endNode < Params.HIDDEN_START_INDEX && endNode >= Params.OUTPUT_START_INDEX)) {
				System.out.println("Link between nodes do not exist yet, creating new link");
				linkID = Globals.getInnovationId();
				Globals.INNOVATION_MAP.get(startNode).put(endNode, linkID);
			} else {
				System.out.println("Link is not valid, breaking out");
				return;
			}
		}
		genes.add(new Gene(linkID, startNode, endNode, ((Math.random() * Params.WEIGHT_MUTATION_RANGE * 2) - Params.WEIGHT_MUTATION_RANGE)));
	}

	/**
	 * mutates by creating a node
	 * can mutate a currently disabled node
	 */
	public void mutateNode() {
		Gene chosenGene = genes.get((int) Math.floor(Math.random() * genes.size()));
		genes.remove(chosenGene);
		System.out.println("Mutating new node between node " + chosenGene.from + " and node " + chosenGene.to);
		Integer geneID = Globals.NODE_MAP.get(chosenGene.from).get(chosenGene.to);
		if (geneID == null) { //gene does not exist yet
			System.out.println("node between nodes do not exist yet, creating new node");
			//create new node from parents
			geneID = Globals.getNodeId();
			Globals.NODE_MAP.get(chosenGene.from).put(chosenGene.to, geneID);
			Globals.NODE_MAP.get(chosenGene.to).put(chosenGene.from, geneID);

			//create new links from parent to child
			int linkID1 = Globals.getInnovationId();
			Globals.INNOVATION_MAP.get(chosenGene.from).put(geneID, linkID1);
			int linkID2 = Globals.getInnovationId();
			Globals.INNOVATION_MAP.get(geneID).put(chosenGene.to, linkID2);
		}
		if (geneID > neuronCount)
			neuronCount = geneID;
		genes.add(new Gene(Globals.INNOVATION_MAP.get(chosenGene.from).get(geneID), chosenGene.from, geneID, 1));
		genes.add(new Gene(Globals.INNOVATION_MAP.get(geneID).get(chosenGene.to), geneID, chosenGene.to, chosenGene.weight));
	}

	public static Chromosome createDefaultChromosome() {
		Chromosome c = new Chromosome();
		int hid = Params.HIDDEN_START_INDEX;

		//defaultChromosome ids not set yet
		if (Globals.INNOVATION_MAP.get(0) == null)
			initializeStartingIds();

		// Connect input neurons to a single hidden neuron
		for (int i = 0; i < Params.OUTPUT_START_INDEX; i++) {
			c.genes.add(new Gene(i + 1, i + 1, hid + 1, ((Math.random() * Params.WEIGHT_MUTATION_RANGE * 2) - Params.WEIGHT_MUTATION_RANGE)));
		}
		// Connect single hidden neuron to output neurons
		for (int i = Params.OUTPUT_START_INDEX; i < Params.HIDDEN_START_INDEX; i++) {
			c.genes.add(new Gene(i + 1, hid + 1, i + 1, ((Math.random() * Params.WEIGHT_MUTATION_RANGE * 2) - Params.WEIGHT_MUTATION_RANGE)));
		}

		return c;
	}

	public Chromosome clone() {
		Chromosome c = new Chromosome();
		c.genes.addAll(this.genes);
		c.fitness = this.fitness;
		//TODO set ID maybe?
		return c;
	}

	public static void initializeStartingIds() {
		int hid = Params.HIDDEN_START_INDEX;
		int currentID;
		for (int i = 0; i < Params.OUTPUT_START_INDEX; i++) {
			Globals.getNodeId(); //initializes the node's existence
			currentID = Globals.getInnovationId(); //initializes the link's existence
			Globals.INNOVATION_MAP.get(i + 1).put(hid + 1, currentID);
		}
		Globals.getNodeId(); //initializes hidden's node
		for (int i = Params.OUTPUT_START_INDEX; i < Params.HIDDEN_START_INDEX; i++) {
			Globals.getNodeId(); //initializes the node's existence
		}
		for (int i = Params.OUTPUT_START_INDEX; i < Params.HIDDEN_START_INDEX; i++) {
			currentID = Globals.getInnovationId(); //initializes the link's existence
			Globals.INNOVATION_MAP.get(hid + 1).put(i + 1, currentID);
		}
	}


	/**
	 * computes distance between genes compared to another chromosome
	 * used for species placement
	 *
	 * @return
	 */
	public double computeGeneDistance(Chromosome other) {
		double distance = 0;
		double NormalizeValue = Math.max(genes.size(), other.genes.size());
		Collections.sort(genes);
		Collections.sort(other.genes);
		double largestDisjointValue = Math.min(genes.get(genes.size() - 1).id, other.genes.get(other.genes.size() - 1).id);
		double totalWeightDifferenceOfMatchingGenes = 0;
		double numberOfDisjointGenes = 0;
		double numberOfMatchingGenes = 0;
		double numberOfExcessGenes = 0;

		// TODO proper algorithm. I was braindead and did the not efficient one
		boolean matchFound;
		for (Gene gene1 : genes) {
			matchFound = false;
			for (Gene gene2 : other.genes) { //check if it has a match
				if (gene1.id == gene2.id) {
					totalWeightDifferenceOfMatchingGenes += Math.abs(gene1.weight - gene2.weight);
					numberOfMatchingGenes++;
					matchFound = true;
					break;
				}
				if (gene1.id < gene2.id) { //too far to have a match
					break;
				}
			}
			//no match
			if (!matchFound) {
				if (gene1.id <= largestDisjointValue) { //disjoint value
					numberOfDisjointGenes++;
				} else { //excess value
					numberOfExcessGenes++;
				}
			}
		}

		//do the same for gene2, except it doesn't need to add to weight difference
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
				if (gene1.id <= largestDisjointValue) { //disjoint value
					numberOfDisjointGenes++;
				} else { //excess value
					numberOfExcessGenes++;
				}
			}
		}

		distance += Params.C1 * numberOfExcessGenes / NormalizeValue;
		distance += Params.C2 * numberOfDisjointGenes / NormalizeValue;
		distance += Params.C3 * totalWeightDifferenceOfMatchingGenes / numberOfMatchingGenes;
		//System.out.println("geneDistance: " + distance + ", Chrom 1: " + id + ", Chrom 2: " + other.id);
		return distance;
	}

	public double getFitness() {
	    return this.fitness;
    }
}

// TODO formatting for fittest chromosome
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
}

/**
 * Evaluates the fitness task for that chromosome a number of times equal to Params.FITNESS_EVALUATIONS,
 * then returns the average of those fitness evaluations.
 *
 * Use by passing in the chromosome, then call create subtasks which will spawn the required
 * number of worker threads to do the simulation.
 */
class EvaluateChromosomeFitnessTask extends RecursiveTask<Double> {
	private Chromosome chromosome;
	private boolean isSubTask;
	public EvaluateChromosomeFitnessTask(Chromosome chromosome) {
		this.chromosome = chromosome;
		this.isSubTask = false;
	}
	public EvaluateChromosomeFitnessTask(Chromosome chromosome, boolean isSubTask) {
		this.chromosome = chromosome;
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
			return evaluateChromosomeFitness();
		}
	}

	private Collection<EvaluateChromosomeFitnessTask> createSubtasks() {
		List<EvaluateChromosomeFitnessTask> dividedTasks = new ArrayList<>();
		for (int i=0; i<Params.FITNESS_EVALUATIONS; i++)
			dividedTasks.add(new EvaluateChromosomeFitnessTask(chromosome, true));
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
			moves += 1;
		}

		double fitness = (double) s.getRowsCleared();
		fitness = fitness == 0 ? moves / 100.0 : fitness;
		//System.out.printf("Chromosome #%d fitness computed with thread %s%n", chromosome.id, Thread.currentThread().getName());
		return fitness;
	}
}

class EvaluatePopulationFitnessTask extends RecursiveAction {
	private List<Chromosome> population;
	private Chromosome chromosome;
	private boolean isSubTask;

	public EvaluatePopulationFitnessTask(List<Chromosome> population) {
		this.population = population;
		this.isSubTask = false;
	}
	public EvaluatePopulationFitnessTask(Chromosome chromosome, boolean isSubTask) {
		this.chromosome = chromosome;
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
		List<EvaluatePopulationFitnessTask> dividedTasks = new ArrayList<EvaluatePopulationFitnessTask>();
		for (Chromosome c: population) {
			dividedTasks.add(new EvaluatePopulationFitnessTask(c, true));
		}
		return dividedTasks;
	}

	private void evaluateChromosomeFitness() {
		chromosome.fitness = (new EvaluateChromosomeFitnessTask(chromosome).compute());
	}
}

class Params {
	public static final int INPUT_SIZE = State.ROWS*State.COLS+State.N_PIECES;
	public static final int OUTPUT_SIZE = 4+State.COLS;
	public static final int BIAS_START_INDEX = 0;
	public static final int INPUT_START_INDEX = 1;
	public static final int OUTPUT_START_INDEX = INPUT_START_INDEX + INPUT_SIZE;
	public static final int HIDDEN_START_INDEX = OUTPUT_START_INDEX + OUTPUT_SIZE;
	public static final int GENERATION_LIMIT = 20; //Number of iterations
	public static final double FITNESS_LIMIT = 1000; //Value for which we automatically end the search

	public static final int FITNESS_EVALUATIONS = 20; // Number of evaluations performed per chromosome to be averaged
	public static final int POPULATION_SIZE = 100; // Population Size
	public static final double SURVIVAL_THRESHOLD = 0.2; // Percentage of species allowed to survive and breed
	public static final double MAXIMUM_STAGNATION = 20; // Generations of non-improvement before species is culled
	public static final double WEIGHT_MUTATION_RANGE = 2.5; // Range at which the weight can be increased or decreased by
	public static final double WEIGHT_MUTATION_CHANCE = 0.25; // Chance of weight of gene being changed
	public static final double NODE_MUTATION_CHANCE = 0.30; // Chance of inserting a new node
	public static final double LINK_MUTATION_CHANCE = 0.25; // Chance of inserting a new link
	public static final double DISABLE_MUTATION_CHANCE = 0.04; // Chance of a gene being disabled
	public static final double ENABLE_MUTATION_CHANCE = 0.02; // Chance of a gene being enabled
	public static final double CROSSOVER_CHANCE = 0.05; // Chance of interspecies breeding
	public static final double COMPATIBILITY_THRESHOLD = 10; // Threshold for measuring species compatibility
	public static final double C1 = 1; // Coefficient for importance of disjoint genes in measuring compatibility
	public static final double C2 = 1; // Coefficient for excess genes
	public static final double C3 = 3; // Coefficient for average weight difference
}

class Globals {
	//TODO change 'node' into 'neuron'
	public static int NEURON_COUNT = Params.HIDDEN_START_INDEX+1;
	public static int INNOVATION_COUNT = 0;
	public static int CHROMOSOME_COUNT = 0;
	public static int NODE_COUNT = 0;
	public static int SPECIES_COUNT = 0;

	//map is (start reference node, target reference node, id)
	public static Map<Integer, Map<Integer, Integer>> INNOVATION_MAP = new HashMap<Integer, Map<Integer, Integer>>();
	//map is (first parent reference node, second parent reference node, id)
	public static Map<Integer, Map<Integer, Integer>> NODE_MAP = new HashMap<Integer, Map<Integer, Integer>>();

	public static int getInnovationId() {
		INNOVATION_COUNT++;
		return INNOVATION_COUNT;
	}

	public static int getChromosomeId() {
		CHROMOSOME_COUNT++;
		return CHROMOSOME_COUNT;
	}

	public static int getSpeciesId() {
		SPECIES_COUNT++;
		return SPECIES_COUNT;
	}

	public static int getNodeId() {
		NODE_COUNT++;
		NODE_MAP.put(NODE_COUNT, new HashMap<Integer, Integer>());
		INNOVATION_MAP.put(NODE_COUNT, new HashMap<Integer, Integer>());
		return NODE_COUNT;
	}
}

/**
 * Handles running an experiment
 */
abstract class Experiment {
    protected Population pop;
    protected Parameters params;

    /**
     * @param inputSize Size of input of the neural network
     * @param outputSize Size of output of the neural network
     * @param hiddenSize Default size of the hidden nodes
     */
    public Experiment(int inputSize, int outputSize, int hiddenSize) {
        this.params = new Parameters(inputSize, outputSize, hiddenSize);
        this.pop = new Population(params,
                this::createDefaultChromosome,
                this::evaluateChromosomeFitness);
    }

    /**
     * Runs the experiment until fitness limit or generation limit is reached
     */
    public void run() {
        while (pop.getFittestChromosome().getFitness() < params.FITNESS_LIMIT &&
                pop.getGeneration() < params.GENERATION_LIMIT) {
            pop.advance();
        }
    }

    abstract public Chromosome createDefaultChromosome();
    abstract public double evaluateChromosomeFitness(Chromosome chromosome);
}

class XORExperiment extends Experiment {
    public XORExperiment() {
        super(2, 1, 0);
        this.params.FITNESS_LIMIT = 1;
        this.params.GENERATION_LIMIT = 100;
    }

    @Override
    public Chromosome createDefaultChromosome() {
        // Connect bias and input nodes to output nodes
        ArrayList<Gene> genes = new ArrayList<>();
        for (int i=params.BIAS_START_INDEX; i<params.OUTPUT_START_INDEX; i++) {
            for (int o=params.OUTPUT_START_INDEX; i<params.HIDDEN_START_INDEX; o++) {
                genes.add(pop.innovator.innovateLink(i, o));
            }
        }

        // Create the default chromosome
        Chromosome c = new Chromosome(this.pop, genes, params.NETWORK_SIZE);
        return c;
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
            outputs = (Double[]) nn.activate(Arrays.asList(c[0])).toArray();
            error += Math.pow(c[1][0] - outputs[0], 2);
        }

        return 1.0 - error;
    }
}

class Parameters {
    public int INPUT_SIZE = State.ROWS*State.COLS+State.N_PIECES;
    public int OUTPUT_SIZE = 4+State.COLS;
    public int HIDDEN_SIZE = 0;
    public int NETWORK_SIZE = 1 + INPUT_SIZE + OUTPUT_SIZE + HIDDEN_SIZE;
    public int BIAS_START_INDEX = 0;
    public int INPUT_START_INDEX = 1;
    public int OUTPUT_START_INDEX = INPUT_START_INDEX + INPUT_SIZE;
    public int HIDDEN_START_INDEX = OUTPUT_START_INDEX + OUTPUT_SIZE;
    public int GENERATION_LIMIT = 20; //Number of iterations
    public double FITNESS_LIMIT = 1000; //Value for which we automatically end the search

    public int FITNESS_EVALUATIONS = 20; // Number of evaluations performed per chromosome to be averaged
    public int POPULATION_SIZE = 100; // Population Size
    public double SURVIVAL_THRESHOLD = 0.2; // Percentage of species allowed to survive and breed
    public double MAXIMUM_STAGNATION = 20; // Generations of non-improvement before species is culled
    public double WEIGHT_MUTATION_RANGE = 2.5; // Range at which the weight can be increased or decreased by
    public double WEIGHT_MUTATION_CHANCE = 0.25; // Chance of weight of gene being changed
    public double NODE_MUTATION_CHANCE = 0.30; // Chance of inserting a new node
    public double LINK_MUTATION_CHANCE = 0.25; // Chance of inserting a new link
    public double DISABLE_MUTATION_CHANCE = 0.04; // Chance of a gene being disabled
    public double ENABLE_MUTATION_CHANCE = 0.02; // Chance of a gene being enabled
    public double CROSSOVER_CHANCE = 0.05; // Chance of interspecies breeding
    public double COMPATIBILITY_THRESHOLD = 10; // Threshold for measuring species compatibility
    public double C1 = 1; // Coefficient for importance of disjoint genes in measuring compatibility
    public double C2 = 1; // Coefficient for excess genes
    public double C3 = 3; // Coefficient for average weight difference

    public Parameters(int inputSize, int outputSize, int hiddenSize) {
        this.INPUT_SIZE = inputSize;
        this.OUTPUT_SIZE = outputSize;
        this.HIDDEN_SIZE = hiddenSize;
        this.NETWORK_SIZE = 1 + inputSize + outputSize + hiddenSize;
        this.OUTPUT_START_INDEX = INPUT_START_INDEX + INPUT_SIZE;
        this.HIDDEN_START_INDEX = OUTPUT_START_INDEX + OUTPUT_SIZE;
    }
}

class Population {
    private int chromosomeCount;
    private int generation;
    private Parameters params;
    private Innovator innovator;
    private List<Chromosome> chromosomes;
    private Function<Chromosome, Double> chromosomeFitnessEvaluator;

    public Population(Parameters params,
                      Supplier<Chromosome> defaultChromosomeCreator,
                      Function<Chromosome, Double> chromosomeFitnessEvaluator) {
        this.chromosomeCount = 0;
        this.chromosomes = new ArrayList<>(params.POPULATION_SIZE);
        this.params = params;
        this.innovator = new Innovator(params.NETWORK_SIZE);
        this.chromosomeFitnessEvaluator = chromosomeFitnessEvaluator;
        this.generation = 0;
        this.populate(defaultChromosomeCreator.get());
    }

    /**
     * Populate the population by copying the base chromosome and mutating them
     * @param base Base chromosome to make copies of
     */
    private void populate(Chromosome base) {
        for (int i=0; i<params.POPULATION_SIZE; i++)
            chromosomes.add((new Chromosome(base)).mutate());
    }

    /**
     * @return A new unique chromosome ID
     */
    private int getNewChromosomeId() {
        return chromosomeCount++;
    }

    /**
     * Advance the population to the next generation
     */
    public void advance() {

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
        Integer[] key = new Integer[]{from, to};
        if (!linkInnovations.containsKey(key))
            linkInnovations.put(key, new Gene(getNewInnovationId(), from, to));
        return new Gene(linkInnovations.get(key), true);
    }

    /**
     * Create a new node innovation if it has yet to exist and clone it
     * @param from Neuron giving output
     * @param to Neuron receiving input
     * @return 2 new genes connecting the neurons to a hidden neuron
     */
    public Gene innovateNode(int from, int to) {
        Integer[] key = new Integer[]{from, to};
        if (!nodeInnovations.containsKey(key))
            nodeInnovations.put(key, new Gene[]{
                    new Gene(getNewInnovationId(), from, getNewNeuronId()),
                    new Gene(getNewInnovationId(), getNewNeuronId(), to),
            });
        return new Gene(linkInnovations.get(key), true);
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
