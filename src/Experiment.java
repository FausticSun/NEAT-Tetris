import java.util.logging.Logger;

public class Experiment {
    private static final Logger LOGGER = Logger.getLogger(Experiment.class.getName());
    private Parameters params;
    private Population population;

    public Experiment(Parameters params) {
        this.params = params;
        this.population = new Population(params);
    }

    public void run(int genNum) {
        LOGGER.info(String.format("Starting the experiment at generation %d", population.getGeneration()));
        if (genNum == 0) {
            while (population.getFittestChromosome().getFitness() < params.FITNESS_LIMIT &&
                    population.getGeneration() < params.GENERATION_LIMIT) {
                population.advance();
            }
        } else {
            for (int i=0; i<genNum; i++) {
                population.advance();
            }
        }
    }

    public Chromosome getFittest() {
        return population.getFittestChromosome();
    }

    public int getGeneration() {
        return population.getGeneration();
    }

    public Parameters getParams() {
        return params;
    }
}
