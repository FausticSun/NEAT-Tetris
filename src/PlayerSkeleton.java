import java.util.ArrayList;
import java.util.Iterator;
import java.util.logging.Logger;

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
            ex.run(1);

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
        }
    }
}