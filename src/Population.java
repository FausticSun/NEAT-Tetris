import java.util.*;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class Population {
    private static final Logger LOGGER = Logger.getLogger( Population.class.getName() );
    private List<Species> species = new ArrayList<>();
    private Innovator innovator;
    private Parameters params;
    private IdGenerator chromosomeIdGenerator = new IdGenerator();
    private IdGenerator speciesIdGenerator = new IdGenerator();
    private int generation = 0;
    private int stagnation = 0;
    private double bestFitness = 0;
    private double compatibility;
    private double compatExponent;

    public Population(Parameters params) {
        this.params = params;
        this.innovator = new Innovator(params);
        this.compatExponent = params.COMPATIBILITY_THRESHOLD;
        this.compatibility = Math.exp(compatExponent);
        generateNewPopulation();
    }

    private void generateNewPopulation() {
        LOGGER.info(String.format("Generating new population"));
        Chromosome defaultChromosome = new Chromosome(params, innovator, chromosomeIdGenerator);
        List<Chromosome> offsprings = new ArrayList<>();
        for (int i=0; i<params.POPULATION_SIZE; i++)
            offsprings.add((new Chromosome(defaultChromosome)).mutateAllGenesWeight().mutate());
        evaluatePopulationFitness(offsprings);
        allocateOffspringsToSpecies(offsprings);
        setStagnation();
        dynamicThresholding();
        getFittestChromosome().save();
        LOGGER.info(String.format("Generation %d best fitness: %f",
                generation,
                this.getFittestChromosome().getFitness()));
    }

    private void dynamicThresholding() {
        long activeSpecies = species.stream().filter(s -> !s.isEmpty()).count();
        if (activeSpecies < params.TARGET_SPECIES) {
            this.compatExponent -= params.COMPAT_MOD;
            LOGGER.fine(String.format("Decreasing threshold to e^%f", compatExponent));
        } else if (activeSpecies > params.TARGET_SPECIES) {
            this.compatExponent += params.COMPAT_MOD;
            LOGGER.fine(String.format("Increasing threshold to e^%f", compatExponent));
        }
        this.compatibility = Math.exp(compatExponent);
    }

    public void advance() {
        this.generation++;
        LOGGER.info(String.format("Entering generation %d", this.generation));
        generateNextPopulation();
        LOGGER.info(String.format("Generation %d best Fitness: %f, gene count: %d, node count: %d",
                generation,
                this.getFittestChromosome().getFitness(),
                this.getFittestChromosome().getGenes().size(),
                this.getFittestChromosome().getGenes().stream()
                        .mapToInt(Link::getFrom)
                        .distinct().count()));
    }

    private void generateNextPopulation() {
        innovator.clear();
        pruneStagnantSpecies();
        List<Chromosome> offsprings = generateOffsprings();
        LOGGER.info(String.format("%d new offsprings generated", offsprings.size()));
        evaluatePopulationFitness(offsprings);
        clearSpeciesChromosomes();
        allocateOffspringsToSpecies(offsprings);
        setStagnation();
        dynamicThresholding();
        getFittestChromosome().save();
    }

    private void evaluatePopulationFitness(List<Chromosome> offsprings) {
        offsprings.stream()
                .forEach(Chromosome::evaluateFitness);
    }

    private void pruneStagnantSpecies() {
        long activeSpeciesCount = species.parallelStream()
                .filter(s -> !s.isEmpty())
                .count();
        if (activeSpeciesCount == 1) {
            this.resetStagnation();
            species.get(0).resetStagnation();
        } else if (this.isStagnant()) {
            this.resetStagnation();
            species = species.stream()
                    .filter(s -> !s.isEmpty())
                    .sorted(Comparator.reverseOrder())
                    .limit(2)
                    .collect(Collectors.toList());
            for (Species specie: species) {
                specie.resetStagnation();
            }
        } else {
            species.removeIf(Species::isStagnant);
        }
    }

    private void resetStagnation() {
        this.stagnation = 0;
    }

    private void allocateOffspringsToSpecies(List<Chromosome> offsprings) {
        for (Chromosome offspring: offsprings) {
            found:
            {
                Species hint = species.stream()
                        .filter(s -> s.getId() == offspring.getSpeciesHint())
                        .findFirst().orElse(null);
                if (hint != null && hint.addChromosome(offspring, compatibility)) {
                    break found;
                }
                for (Species specie : species) {
                    if (specie.addChromosome(offspring, compatibility)) {
                        break found;
                    }
                }
                Species newSpecies = new Species(params, this, speciesIdGenerator.getNextId(), offspring);
                species.add(newSpecies);
                LOGGER.fine(String.format("Species not found, creating S%d from C%d",
                        newSpecies.getId(), offspring.getId()));
            }
        }
        LOGGER.info(String.format("Total Species - %d, Active Species = %d, Threshold - %f",
                species.size(),
                species.stream().filter(s -> !s.isEmpty()).count(),
                compatibility));
        for (Species specie: species)
            specie.setStagnation();
    }

    private void clearSpeciesChromosomes() {
        for (Species specie: species)
            specie.clear();
    }

    private List<Chromosome> generateOffsprings() {
        List<Chromosome> offsprings = new ArrayList<>();
        List<Species> activeSpecies = species.stream().filter(s -> !s.isEmpty()).collect(Collectors.toList());
        double speciesFitnessAverageSum = activeSpecies.stream().mapToDouble(Species::getAverageFitness).sum();
        for (Species specie: activeSpecies) {
            int allocatedOffsprings = (int) ((specie.getAverageFitness()/speciesFitnessAverageSum)
                    *params.POPULATION_SIZE);
            specie.cull();
            offsprings.addAll(specie.generateOffsprings(allocatedOffsprings));
        }
        return offsprings;
    }

    public int getGeneration() {
        return generation;
    }

    public Chromosome getFittestChromosome() {
        return species.stream()
                .filter(s -> !s.isEmpty())
                .map(Species::getFittestChromosome)
                .max(Comparator.naturalOrder())
                .get();
    }

    private void setStagnation() {
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

    private boolean isStagnant() {
        return this.stagnation >= params.MAXIMUM_POPULATION_STAGNATION;
    }

    public Chromosome getRandomChromosome() {
        List<Chromosome> randomChromosomes = this.species.stream()
                .filter(s -> !s.isEmpty())
                .map(Species::getRandomChromosome)
                .collect(Collectors.toList());
        return randomChromosomes.get((new Random()).nextInt(randomChromosomes.size()));
    }

    public void savePopulation(List<Chromosome> pop) {
        pop.stream().forEach(Chromosome::save);
    }
}
