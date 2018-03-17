import java.awt.event.WindowEvent;
import java.util.List;
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
            ex.run(5);

            LOGGER.info(String.format("Demoing fittest of Generation %d", ex.getGeneration()));
            s = new TetrisState();
            demo = new TFrame(s);
            nn = new NeuralNet(ex.getFittest().getNeuralNet());

            while (!s.hasLost()) {
                s.setOutputs(nn.activate(s.getInputs()));

                s.draw();
                s.drawNext(0, 0);
                try {
                    Thread.sleep(300);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            demo.dispose();
            LOGGER.info(String.format("%d moves made with %d rows cleared", s.getTurnNumber(), s.getRowsCleared()));
        }
    }
}