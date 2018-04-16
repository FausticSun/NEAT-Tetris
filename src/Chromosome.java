import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.function.Function;
import java.util.logging.Logger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class Chromosome implements Comparable<Chromosome>, Saveable {
    private static final Logger LOGGER = Logger.getLogger( Chromosome.class.getName() );
    private Parameters params;
    private Innovator innovator;
    private IdGenerator idGenerator;
    private int id;
    private List<Gene> genes = new ArrayList<>();
    private double fitness = -1;
    private int speciesHint = 0;

    public Chromosome(Parameters params, Innovator innovator, IdGenerator idGenerator) {
        this.params = params;
        this.innovator = innovator;
        this.idGenerator = idGenerator;
        this.id = idGenerator.getNextId();
        for (Link link: params.DEFAULT_CHROMOSOME_BLUEPRINT)
            genes.add(innovator.innovateLink(link));
    }

    public Chromosome(Chromosome o, List<Gene> newGenes) {
        this.params = o.params;
        this.innovator = o.innovator;
        this.idGenerator = o.idGenerator;
        this.id = idGenerator.getNextId();
        this.speciesHint = o.speciesHint;
        for (Gene gene: newGenes)
            genes.add(new Gene(gene));
        LOGGER.fine(String.format("Creating new chromosome C%d", id));
    }

    public Chromosome(Chromosome o) {
        this.params = o.params;
        this.innovator = o.innovator;
        this.idGenerator = o.idGenerator;
        this.id = idGenerator.getNextId();
        this.speciesHint = o.speciesHint;
        for (Gene gene: o.genes)
            genes.add(new Gene(gene));
        LOGGER.fine(String.format("Creating new chromosome C%d", id));
    }

    public int getId() {
        return id;
    }

    public void evaluateFitness() {
        NeuralNet nn = new NeuralNet(params, this);
        fitness = IntStream.range(0, 5).mapToDouble(i -> {
            TetrisState s = new TetrisState(nn);
            while (!s.hasLost())
                s.makeBestMove();
            return s.getFitness();
        }).average().orElse(0);
        LOGGER.fine(String.format("C%d evaluated fitness: %f", getId(), fitness));
    }

    @Override
    public int compareTo(Chromosome o) {
        return Double.compare(this.getFitness(), o.getFitness());
    }
    
    public static Chromosome deserialize(Parameters param, Innovator innovator, IdGenerator idg, String data) {
        Chromosome c = new Chromosome(param, innovator, idg);
        c.genes.clear();
        String[] lines = data.split(";");
        for (String aGene : lines) {
            if(!aGene.trim().equals("")) {
                String[] geneData = aGene.split(",");
                int id = Integer.parseInt(geneData[0]);
                int from = Integer.parseInt(geneData[1]);
                int to = Integer.parseInt(geneData[2]);
                double weight = Double.parseDouble(geneData[3]);
                c.genes.add(new Gene(param, id, from, to, weight, geneData[4].equals("t")));
            }
        }
        return c;
    }

    public double getFitness() {
        return fitness;
    }
    
    public Parameters getParams() {
    	return params;
	}

    public Chromosome mutate() {
        if (params.CLASSIC_TOPOLOGY_MUTATION) {
            double randomNumber = Math.random();
            mutated:
            {
                if (randomNumber < params.LINK_MUTATION_CHANCE) {
                    mutateLink();
                    break mutated;
                }
                randomNumber -= params.LINK_MUTATION_CHANCE;
                if (randomNumber < params.NODE_MUTATION_CHANCE) {
                    mutateNode();
                    break mutated;
                }
                randomNumber -= params.NODE_MUTATION_CHANCE;
                if (randomNumber < params.WEIGHT_MUTATION_CHANCE) {
                    mutateGeneWeight();
                    break mutated;
                }
                randomNumber -= params.WEIGHT_MUTATION_CHANCE;
                if (randomNumber < params.DISABLE_MUTATION_CHANCE) {
                    mutateGeneToggle(true);
                    break mutated;
                }
                randomNumber -= params.DISABLE_MUTATION_CHANCE;
                if (randomNumber < params.ENABLE_MUTATION_CHANCE) {
                    mutateGeneToggle(false);
                    break mutated;
                }
            }
        } else {
            anjiMutate();
        }
        
        return this;
    }

    private void anjiMutate() {
        anjiMutateGenes();
        anjiMutateLink();
        anjiMutateNode();
    }

    private void anjiMutateNode() {
        Gene g;
        for (int i=0; i<genes.size(); i++) {
            g = genes.get(i);
            if (g.isEnabled() && Math.random() < params.NODE_MUTATION_CHANCE) {
                g.toggleEnabled();
                genes.addAll(innovator.innovateNode(g));
            }
        }
    }

    private void anjiMutateLink() {
        List<Link> potentialMutations = khanTopologicalSort();
        for(Link l : potentialMutations) {
        	anjiMutateLink(l.getFrom(), l.getTo());
		}
    }

    private void anjiMutateLink(int from, int to) {
        if (Math.random() < params.LINK_MUTATION_CHANCE) {
            genes.add(innovator.innovateLink(new Link(from, to)));
        }
    }

    private boolean isHidden(int i) {
        return !isInput(i) && !isOutput(i);
    }

    private boolean isDisabled(int i) {
        return false;
    }

    private boolean isOutput(int i) {
        return i >= params.OUTPUT_START_INDEX && i <= params.HIDDEN_START_INDEX;
    }

    private boolean isInput(int i) {
        return i >= params.BIAS_START_INDEX && i <= params.OUTPUT_START_INDEX;
    }

    private void anjiMutateGenes() {
        for (Gene g: genes) {
            if ((g.isEnabled() && Math.random() < params.DISABLE_MUTATION_CHANCE) ||
                    (!g.isEnabled() && Math.random() < params.ENABLE_MUTATION_CHANCE)) {
                g.toggleEnabled();
            }
            if (Math.random() < params.WEIGHT_MUTATION_CHANCE) {
                g.perterbWeight();
            }
        }
    }

    private void mutateGeneToggle(boolean b) {
        List<Gene> potentialGenes = genes.stream()
                .filter(g -> g.isEnabled() == b)
                .collect(Collectors.toList());
        if (potentialGenes.size() > 0) {
            potentialGenes.get((new Random().nextInt(potentialGenes.size()))).toggleEnabled();
        }
    }

    private void mutateGeneWeight() {
        List<Gene> potentialGenes = genes.stream()
                .filter(Gene::isEnabled)
                .collect(Collectors.toList());
        if (potentialGenes.size() > 0) {
            potentialGenes.get((new Random().nextInt(potentialGenes.size()))).perterbWeight();
        }
    }

    private void mutateNode() {
        List<Gene> potentialGenes = genes.stream()
                .filter(Gene::isEnabled)
                .collect(Collectors.toList());
        if (potentialGenes.isEmpty()) {
            return;
        }
        Gene chosenGene = potentialGenes.get((new Random().nextInt(potentialGenes.size())));
        chosenGene.toggleEnabled();
        genes.addAll(innovator.innovateNode(chosenGene));
    }

    private void mutateLink() {
        Link newLink = getNewLink();
        if (newLink != null) {
            genes.add(innovator.innovateLink(newLink));
        }
    }

    private Link getNewLink() {
        // Get a list of neurons present in this chromosome
        Set<Integer> presentNeuronsSet = new TreeSet<>();
        // Add bias, input and output neurons
        for (int i=0; i<=params.HIDDEN_START_INDEX; i++) {
            presentNeuronsSet.add(i);
        }
        for (Gene g: genes) {
            presentNeuronsSet.add(g.getFrom());
            presentNeuronsSet.add(g.getTo());
        }
        List<Integer> presentNeuronsList = new ArrayList<>(presentNeuronsSet);

        // Attempt to get 2 neurons that fulfill all requirements
        int from, to;
        boolean isExist;
        for (int i=0; i<100; i++) {
            // Get 2 neurons
            from = presentNeuronsList.get((new Random()).nextInt(presentNeuronsList.size()));
            to = presentNeuronsList.get((new Random()).nextInt(presentNeuronsList.size()));
            // Check if the neurons are the same
            if (from == to)
                continue;
            // Check if the neurons are 2 inputs
            if (from < params.OUTPUT_START_INDEX && to < params.OUTPUT_START_INDEX)
                continue;
            // Check if the neurons are 2 outputs
            if (from >= params.OUTPUT_START_INDEX && from < params.HIDDEN_START_INDEX &&
                    to >= params.OUTPUT_START_INDEX && from < params.HIDDEN_START_INDEX)
                continue;
            // Check if a link already exists
            isExist = false;
            for (Gene g: genes) {
                if ((g.getFrom() == from && g.getTo() == to) ||
                        (g.getFrom() == to && g.getTo() == from)) {
                    isExist = true;
                }
            }
            if (isExist)
                continue;
            // If 2 neurons are hidden, perform DFS to determine from and to
            if (to >= params.HIDDEN_START_INDEX && from >= params.HIDDEN_START_INDEX) {
                if (!dfs(from, to)) {
                    int temp = from;
                    from = to;
                    to = temp;
                }
            }
            // Else, make sure input neuron is from or output neuron is to
            else if (to < params.OUTPUT_START_INDEX ||
                    (from >= params.OUTPUT_START_INDEX && from < params.HIDDEN_START_INDEX)) {
                int temp = from;
                from = to;
                to = temp;
            }
            // Make the link
            return new Link(from, to);
        }
        return null;
    }

    private boolean dfs(int from, int to) {
        Set<Integer> visited = new TreeSet<>();
        Stack<Integer> stack = new Stack<>();
        stack.push(from);
        while (!stack.empty()) {
            final int v = stack.pop();
            if (!visited.contains(v)) {
                visited.add(v);
                for (int i: genes.stream()
                        .filter(g -> g.getFrom() == v)
                        .mapToInt(Link::getTo)
                        .toArray()) {
                    if (i == to)
                        return true;
                    stack.push(i);
                }
            }
        }
        return false;
    }
    
    private List<Link> khanTopologicalSort() {
    	//Generate Adj list
		Map<Integer, List<Gene>> adjList = new HashMap<>();
		Map<Integer, Integer> incomingCounts = new HashMap<>();
		for (Gene g : genes) {
			if(!adjList.containsKey(g.getFrom())) {
				adjList.put(g.getFrom(), new ArrayList<>());
			}
			adjList.get(g.getFrom()).add(g);
			if(!incomingCounts.containsKey(g.getTo())) {
				incomingCounts.put(g.getTo(), 0);
			}
			incomingCounts.put(g.getTo(), incomingCounts.get(g.getTo()) + 1);
		}
		
		//Ensure that incomingCounts contain all the output nodes and are never removed.
		for(int i = params.OUTPUT_START_INDEX; i < params.HIDDEN_START_INDEX; i++) {
			if(!incomingCounts.containsKey(i)) {
				incomingCounts.put(i, 1);
			} else {
				incomingCounts.put(i, incomingCounts.get(i) + 1);
			}
		}
		
		//Start by processing all the input nodes
		Queue<Integer> inputItemsToProcess = new LinkedBlockingQueue<>();
		for (int i=params.BIAS_START_INDEX; i<params.OUTPUT_START_INDEX; i++) {
			inputItemsToProcess.add(i);
		}
		
		//After that, process these nodes
		Queue<Integer> itemsToProcess = new LinkedBlockingQueue<>();
		
		List<Link> results = new ArrayList<>();
		
		while(inputItemsToProcess.size() > 0 || itemsToProcess.size() > 0) {
			int nextItem;
			if(inputItemsToProcess.size() > 0) {
				nextItem = inputItemsToProcess.poll();
			} else {
				nextItem = itemsToProcess.poll();
			}
			
			//Add all remaining unprocessed nodes as a potential link from what we have to there
			incomingCounts.forEach((key, value) -> results.add(new Link(nextItem, key)));
			itemsToProcess.forEach(integer -> results.add(new Link(nextItem, integer)));
			
			//Process this node, removing it's links from all next nodes.
			List<Gene> outgoings = new ArrayList<>();
			if(adjList.containsKey(nextItem)) {
				outgoings = adjList.get(nextItem);
				adjList.remove(nextItem);
			}
			
			//Add nodes with no remaining input links to itemsToProcess.
			for(Gene outgoingGene : outgoings) {
				int newOutgoing = incomingCounts.get(outgoingGene.getTo()) - 1;
				if(newOutgoing == 0) {
					incomingCounts.remove(outgoingGene.getTo());
					itemsToProcess.add(outgoingGene.getTo());
				} else {
					incomingCounts.put(outgoingGene.getTo(), newOutgoing);
				}
			}
		}
	
		return results;
	}

    public Chromosome mutateAllGenesWeight() {
        for (Gene g: genes) {
            g.perterbWeight();
        }
        return this;
    }

    public Chromosome breedWith(Chromosome other) {
        // Ensure that this has a higher fitnessFuture than other.
        if(other.getFitness() > this.getFitness())
            return other.breedWith(this);

        // Randomize same genes and add other genes from fitter parent
        Function<Gene, Gene> randomizeSameGenes = (g) -> {
            Gene otherGene = other.genes.stream()
                    .filter(o -> (g.getId() == o.getId()))
                    .findFirst()
                    .orElse(g);
            return Math.random() < 0.5 ? g : otherGene;
        };
        List<Gene> newGenes;
        newGenes = this.genes.parallelStream()
                .map(randomizeSameGenes)
                .collect(Collectors.toList());

        // Add other genes from other parent if fitness is similar
        if (this.similarFitness(other)) {
            newGenes.addAll(other.genes.parallelStream()
                    .filter(s -> this.genes.stream()
                            .noneMatch(o -> o.getId() == s.getId()))
                    .collect(Collectors.toList())
            );
        }

        // Clone child from fitter parent, and pass its genes
        Chromosome child = new Chromosome(this, newGenes);

        // Mutate child
        child.mutate();

        return child;
    }

    private boolean similarFitness(Chromosome other) {
        return Math.abs(this.getFitness() - other.getFitness())
                / Math.max(this.getFitness(), other.getFitness())
                < params.SIMILAR_FITNESS_DISCREPANCY;
    }

    public double distanceFrom(Chromosome other) {
        final int SAME = 0;
        final int DISJOINT = 1;
        final int EXCESS = 2;
        // Calculate normalized value
        double NormalizeValue = Math.max(this.genes.size(), other.genes.size());
        // Check if either chromosome is empty
        if (this.genes.isEmpty() || other.genes.isEmpty()) {
            // Check if both chromosomes are empty
            if (NormalizeValue == 0) {
                return 0;
            }
            return params.EXCESS_COEFFICIENT * (this.genes.size() + other.genes.size()) / NormalizeValue;
        }
        // Initialize distance
        double distance = 0;
        // Compute last innovation split
        int thisMaxId = Collections.max(this.genes).getId();
        int otherMaxId = Collections.max(other.genes).getId();
        int minMaxId = Math.min(thisMaxId, otherMaxId);

        // Partition to same, excess and disjoint
        Function<Gene, Integer> thisGeneClassifier = (g) -> {
            if (other.genes.parallelStream().anyMatch(o -> g.getId() == o.getId())) {
                return SAME;
            }
            if (g.getId() <= minMaxId) {
                return DISJOINT;
            } else {
                return EXCESS;
            }
        };
        Map<Integer, List<Gene>> thisGroupedGenes = this.genes.parallelStream()
                .collect(Collectors.groupingByConcurrent(thisGeneClassifier));
        Function<Gene, Integer> otherGeneClassifier = (g) -> {
            if (thisGroupedGenes.get(SAME) != null &&
                    thisGroupedGenes.get(SAME).parallelStream().anyMatch(o -> g.getId() == o.getId())) {
                return SAME;
            }
            if (g.getId() <= minMaxId) {
                return DISJOINT;
            } else {
                return EXCESS;
            }
        };
        Map<Integer, List<Gene>> otherGroupedGenes = this.genes.parallelStream()
                .collect(Collectors.groupingByConcurrent(otherGeneClassifier));

        // Compute sum of weight differences
        double averageWeightDifference = 0;
        if (thisGroupedGenes.get(SAME) != null) {
            thisGroupedGenes.get(SAME).sort(Comparator.naturalOrder());
            otherGroupedGenes.get(SAME).sort(Comparator.naturalOrder());
            averageWeightDifference = IntStream.range(0, thisGroupedGenes.get(SAME).size())
                    .mapToDouble(i -> Math.abs(thisGroupedGenes.get(SAME).get(i).getWeight() -
                            otherGroupedGenes.get(SAME).get(i).getWeight()))
                    .average().orElse(0.0);
        }

        // Compute count of excess and disjoint count
        List<Gene> empty = new ArrayList<>();
        int disjointCount = thisGroupedGenes.getOrDefault(DISJOINT, empty).size() +
                otherGroupedGenes.getOrDefault(DISJOINT, empty).size();
        int excessCount = thisGroupedGenes.getOrDefault(EXCESS, empty).size() +
                otherGroupedGenes.getOrDefault(EXCESS, empty).size();

        distance += params.WEIGHT_DIFFERENCE_COEFFICIENT * averageWeightDifference;
        distance += params.DISJOINT_COEFFICIENT * disjointCount / NormalizeValue;
        distance += params.EXCESS_COEFFICIENT * excessCount / NormalizeValue;
        return distance;
    }

    public List<Gene> getGenes() {
        return Collections.unmodifiableList(genes);
    }

    public int getSpeciesHint() {
        return speciesHint;
    }

    public void setSpeciesHint(int speciesHint) {
        this.speciesHint = speciesHint;
    }
    
    public String toString() {
        return "Chromosome: ID: " + id + ", fitness: " + fitness + ", gene count: " + genes.size();
    }
    
    public String serialize() {
        String result = "";
        for (Gene g: genes) {
            result += g.getId() + "," + g.getFrom() + "," + g.getTo() + "," + g.getWeight() + ",";
            if(!g.isEnabled()) {
                result += "f";
            } else {
                result += "t";
            }
            result += ";";
        }
        return result;
    }
    
    public String createDotData() {
        String result = "digraph G {\n";
    
        for (Gene g : genes) {
            result += "    " + g.getFrom() + " -> " + g.getTo() + " [label=" + g.getWeight();
            if (!g.isEnabled()) {
                result += ",style=dotted";
            }
            result += "];\n";
        }
    
        return result + "\n}";
    }

    public void save() {
        save(String.format("C%d", this.id));
    }

    public void save(String filename) {
        try (PrintWriter pr = new PrintWriter(String.format("Chromosomes/%s", filename))) {
            for (Gene g: genes) {
                pr.println(g.toString());
            }
        } catch (IOException ioe) {
            LOGGER.severe(ioe.getMessage());
        }
    }
}
