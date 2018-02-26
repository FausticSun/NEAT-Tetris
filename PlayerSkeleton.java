import java.util.*;
import java.lang.StringBuilder;

public class PlayerSkeleton {
	static boolean HEADLESS = false;

	public static void main(String[] args) {
		for (String arg : args) {
			switch(arg) {
				case "headless":
					HEADLESS = true; break;
				default:
					break;
			}
		}
		new PlayerSkeleton();
	}

	public PlayerSkeleton() {
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
	public static int[][] maxSlots = new int[State.N_PIECES][];
	static {
		for (int i=0; i<State.N_PIECES; i++) {
			int[][] moves = State.legalMoves[i];
			maxSlots[i] = new int[State.pOrients[i]];
			for (int j=0; j<State.pOrients[i]; j++) {
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
		
		System.out.printf("%d, %d, %d%n", s.nextPiece, slot, maxSlots[s.nextPiece][orient]);
		return slot < maxSlots[s.nextPiece][orient] ? slot : -1;
	}
}

class Params {
	public static final int INPUT_SIZE = State.ROWS*State.COLS+State.N_PIECES;
	public static final int OUTPUT_SIZE = 4+State.COLS;
	public static final int BIAS_START_INDEX = 0;
	public static final int INPUT_START_INDEX = 1;
	public static final int OUTPUT_START_INDEX = INPUT_START_INDEX + INPUT_SIZE;
	public static final int HIDDEN_START_INDEX = OUTPUT_START_INDEX + OUTPUT_SIZE;
}

class Globals {
	public static int NEURON_COUNT = Params.HIDDEN_START_INDEX+1;
	public static int INNOVATION_COUNT = 0;

	public static int getInnovationId() {
		INNOVATION_COUNT++;
		return INNOVATION_COUNT;
	}
}