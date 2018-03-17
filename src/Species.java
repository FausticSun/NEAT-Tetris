import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class Species implements Comparable<Species> {
    private static final Logger LOGGER = Logger.getLogger( Population.class.getName() );
    private int id;
    private Parameters params;
    private Population pop;
    private Chromosome representative;
    private List<Chromosome> chromosomes = new ArrayList<>();
    private int stagnation = 0;
    private double bestFitness = 0;

    public Species(Parameters params, Population pop, int id, Chromosome representative) {
        this.params = params;
        this.id = id;
        this.representative = representative;
        this.pop = pop;
        this.chromosomes.add(representative);
    }

    public void resetStagnation() {
        stagnation = 0;
    }

    public boolean isStagnant() {
        return stagnation >= params.MAXIMUM_SPECIES_STAGNATION;
    }

    public boolean addChromosome(Chromosome offspring, double compatibility) {
        double distance = representative.distanceFrom(offspring);
        if (distance < compatibility) {
            LOGGER.fine(String.format("Distance between C%d and S%d is %f < %f, added",
                    offspring.getId(),
                    this.getId(),
                    distance,
                    compatibility));
            chromosomes.add(offspring);
            offspring.setSpeciesHint(this.id);
            return true;
        }
        LOGGER.fine(String.format("Distance between C%d and S%d is %f > %f, rejected",
                offspring.getId(),
                this.getId(),
                distance,
                compatibility));
        return false;
    }

    public void setStagnation() {
        if (chromosomes.isEmpty()) {
            stagnation++;
            return;
        }
        Chromosome fitChromosome = getFittestChromosome();
        if (isFitnessLessOrSimilar(fitChromosome.getFitness(), this.bestFitness))
            stagnation++;
        else {
            this.bestFitness = fitChromosome.getFitness();
            stagnation = 0;
        }
    }

    private boolean isFitnessLessOrSimilar(double f1, double f2) {
        return f1 < (f2 + f2*params.SIMILAR_FITNESS_DISCREPANCY);
    }

    public boolean isEmpty() {
        return chromosomes.isEmpty();
    }

    public void clear() {
        if (!chromosomes.isEmpty()) {
            representative = chromosomes.get((new Random()).nextInt(chromosomes.size()));
            chromosomes.clear();
        }
    }

    public double getAverageFitness() {
        if (chromosomes.isEmpty()) {
            return 0;
        }
        return chromosomes.stream()
                .sorted(Comparator.naturalOrder())
                .collect(Collectors.averagingDouble(c -> c.getFitness() / chromosomes.size()));
    }

    public List<Chromosome> generateOffsprings(int allocatedOffsprings) {
        List<Chromosome> offsprings = new ArrayList<>();
        if (allocatedOffsprings == 0 || chromosomes.isEmpty()) {
            return offsprings;
        }
        offsprings.add(getFittestChromosome());
        Chromosome parent1, parent2;
        while (offsprings.size() < allocatedOffsprings) {
            parent1 = getRandomChromosome();
            if (Math.random() < params.CROSSOVER_CHANCE) {
                parent2 = pop.getRandomChromosome();
            } else {
                parent2 = getRandomChromosome();
            }
            offsprings.add(parent1.breedWith(parent2));
        }
        return offsprings;
    }

    public Chromosome getFittestChromosome() {
        if (chromosomes.isEmpty()) {
            return null;
        }
        return Collections.max(chromosomes);
    }

    public void cull() {
        chromosomes.sort(Collections.reverseOrder());
        int chromosomesToKeep = Math.max(1 ,(int)(chromosomes.size() * params.SURVIVAL_THRESHOLD));
        chromosomes = chromosomes.subList(0, chromosomesToKeep);
    }

    public Chromosome getRandomChromosome() {
        if (chromosomes.isEmpty())
            return null;
        return chromosomes.get((new Random()).nextInt(chromosomes.size()));
    }

    public int getId() {
        return id;
    }

    @Override
    public int compareTo(Species o) {
        return Double.compare(this.bestFitness, o.bestFitness);
    }
}
