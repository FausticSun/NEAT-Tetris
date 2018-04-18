import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.logging.Logger;

public class PlayerSkeleton {
    private static final Logger LOGGER = Logger.getLogger( PlayerSkeleton.class.getName() );
    static boolean HEADLESS = false;
    static boolean EVOLVE = false;

    public static void main(String[] args) {
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            switch(arg) {
                case "headless":
                    HEADLESS = true; break;
                case "evolve":
                    EVOLVE = true; break;
                case "-demo":
                    Parameters demoParams = Parameters.createTetrisParameters();
                    String demoFile = args[i+1];
                    Chromosome demoChromosome = null;
                    try {
                        demoChromosome = new Chromosome(demoParams, new BufferedReader(new FileReader(demoFile)));
                    } catch (FileNotFoundException e) {
                        LOGGER.severe(e.getMessage());
                    }
                    NeuralNet nn = new NeuralNet(demoParams, demoChromosome);
                    TetrisState s = new TetrisState(nn);
                    TFrame demo = new TFrame(s);

                    while(!s.hasLost()) {
                        s.makeBestMove();
                        s.draw();
                        s.drawNext(0,0);
//                        try {
//                            Thread.sleep(10);
//                        } catch (InterruptedException e) {
//                            e.printStackTrace();
//                        }
                    }
                    demo.dispose();
                    System.out.println("You have completed "+s.getRowsCleared()+" rows.");

                default:
                    break;
            }
        }
        new PlayerSkeleton();
    }

    public PlayerSkeleton() {
        (new File("Chromosomes")).mkdirs();
//        State s = new State();
//        ArrayList<Integer> pieces = new ArrayList<>();
//        while (!s.hasLost()) {
//            pieces.add(s.nextPiece);
//            s.makeMove(0);
//        }
//        s = new State();
//        Iterator<Integer> it = pieces.iterator();
//        while(!s.hasLost()) {
//            if (it.next() != s.nextPiece) {
//                System.out.println("FAILED");
//                break;
//            }
//            s.makeMove(0);
//        }
//        System.out.println("END");
        Parameters params = Parameters.createTetrisParameters();
        Experiment ex = new Experiment(params);
        TetrisState s;
        TFrame demo;
        NeuralNet nn;
        while (ex.getGeneration() < params.GENERATION_LIMIT) {
            ex.run(0);

            LOGGER.info(String.format("Demoing fittest of Generation %d", ex.getGeneration()));
            nn = new NeuralNet(params, ex.getFittest());
            s = new TetrisState(nn);
            demo = new TFrame(s);

            while (!s.hasLost()) {
                s.makeBestMove();
    
                s.draw();
                s.drawNext(0, 0);
            }
            demo.dispose();
            LOGGER.info(String.format("%d moves made with %d rows cleared and %f fitness", s.getTurnNumber(), s.getRowsCleared(), s.getFitness()));

            if(ex.getGeneration() % 5 == 0) {
                LOGGER.fine("Exporting fittest chromosome to file...");
                try (BufferedWriter writer = new BufferedWriter(new FileWriter("FitGen" + ex.getGeneration() + ".txt"));) {
                    writer.write(ex.getFittest().serialize());
                    
                    writer.close();
                } catch (IOException ioe) {
                    LOGGER.severe(ioe.toString());
                }
            }
        }
    }
}