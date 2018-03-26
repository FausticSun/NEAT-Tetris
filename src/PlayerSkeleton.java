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
        Parameters params = Parameters.createTetrisParameters();
        Experiment ex = new Experiment(params);
        TetrisState s;
        TFrame demo;
        NeuralNet nn;
        while (ex.getGeneration() < params.GENERATION_LIMIT) {
            ex.run(0);

            LOGGER.info(String.format("Demoing fittest of Generation %d", ex.getGeneration()));
            nn = new NeuralNet(ex.getFittest().getNeuralNet());
            s = new TetrisState(nn);
            demo = new TFrame(s);

            while (!s.hasLost()) {
                s.makeBestMove();

                s.draw();
                s.drawNext(0, 0);
                try {
                    Thread.sleep(50);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            demo.dispose();
            LOGGER.info(String.format("%d moves made with %d rows cleared", s.getTurnNumber(), s.getRowsCleared()));
        }
    }
}