import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

public class TetrisState extends State {
    private int bestMove;
    private NeuralNet nn;

    public TetrisState(NeuralNet subNn) {
        this.nn = subNn;
    }

    public double evaluateHeuristic(int move) {
        // Create a copy of game field for simulation
        int[][] simulatedField = new int[ROWS][];
        for (int i=0; i<ROWS; i++) {
            simulatedField[i] = Arrays.copyOf(getField()[i], COLS);
        }
        int[] simulatedTop = Arrays.copyOf(getTop(), COLS);

        // Convert single int move to orient and slot
        int orient = legalMoves[nextPiece][move][ORIENT];
        int slot = legalMoves[nextPiece][move][SLOT];

        // Simulate the move
        //height if the first column makes contact
        int height = getTop()[slot]-getpBottom()[nextPiece][orient][0];
        //for each column beyond the first in the piece
        for(int c = 1; c < pWidth[nextPiece][orient];c++) {
            height = Math.max(height,getTop()[slot+c]-getpBottom()[nextPiece][orient][c]);
        }

        //check if game ended
        if(height+getpHeight()[nextPiece][orient] >= ROWS) {
            return 0.0;
        }

        int landingHeight = height+getpHeight()[nextPiece][orient]/2;


        //for each column in the piece - fill in the appropriate blocks
        for(int i = 0; i < pWidth[nextPiece][orient]; i++) {

            //from bottom to top of brick
            for(int h = height+getpBottom()[nextPiece][orient][i]; h < height+getpTop()[nextPiece][orient][i]; h++) {
                simulatedField[h][i+slot] = getTurnNumber();
            }
        }

        //adjust top
        for(int c = 0; c < pWidth[nextPiece][orient]; c++) {
            simulatedTop[slot+c]=height+getpTop()[nextPiece][orient][c];
        }

        int rowsCleared = 0;

        //check for full rows - starting at the top
        for(int r = height+getpHeight()[nextPiece][orient]-1; r >= height; r--) {
            //check all columns in the row
            boolean full = true;
            for(int c = 0; c < COLS; c++) {
                if(simulatedField[r][c] == 0) {
                    full = false;
                    break;
                }
            }
            //if the row was full - remove it and slide above stuff down
            if(full) {
                rowsCleared++;
                //for each column
                for(int c = 0; c < COLS; c++) {

                    //slide down all bricks
                    for(int i = r; i < simulatedTop[c]; i++) {
                        simulatedField[i][c] = simulatedField[i+1][c];
                    }
                    //lower the top
                    simulatedTop[c]--;
                    while(simulatedTop[c]>=1 && simulatedField[simulatedTop[c]-1][c]==0) simulatedTop[c]--;
                }
            }
        }

        List<Double> features = new ArrayList<>();
        // El-Tetris features
        // http://imake.ninja/el-tetris-an-improvement-on-pierre-dellacheries-algorithm/
        features.add((double) landingHeight);
        features.add((double) rowsCleared);
        int rowTransitions = 0;
        for (int row=0; row<ROWS; row++) {
            int prev = simulatedField[row][0];
            for (int col=1; col<COLS; col++) {
                if ((prev == 0 && simulatedField[row][col] != 0) ||
                        (prev != 0 && simulatedField[row][col] == 0)) {
                    rowTransitions++;
                }
                prev = simulatedField[row][col];
            }
        }
        features.add((double) rowTransitions);
        int colTransitions = 0;
        for (int col=0; col<COLS; col++) {
            int prev = simulatedField[0][col];
            for (int row=1; row<ROWS; row++) {
                if ((prev == 0 && simulatedField[row][col] != 0) ||
                        (prev != 0 && simulatedField[row][col] == 0)) {
                    colTransitions++;
                }
                prev = simulatedField[row][col];
            }
        }
        features.add((double) colTransitions);
        int holes = 0;
        for (int col=0; col<COLS; col++) {
            for (int row=simulatedTop[col]; row>=0; row--) {
                if (simulatedField[row][col] == 0) {
                    holes++;
                }
            }
        }
        features.add((double) holes);
        int wellSum = 0;
        boolean prevIsWell, thisIsWell;
        prevIsWell = false;
        for (int row=0; row<ROWS; row++) {
            thisIsWell = simulatedField[row][0] == 0 &&
                    simulatedField[row][1] != 0;
            if (prevIsWell && thisIsWell) {
                wellSum++;
            }
            prevIsWell = thisIsWell;
        }
        for (int col=1; col<COLS-1; col++) {
            prevIsWell = false;
            for (int row=0; row<ROWS; row++) {
                thisIsWell = simulatedField[row][col] == 0 &&
                        simulatedField[row][col-1] != 0 &&
                        simulatedField[row][col+1] != 0;
                if (prevIsWell && thisIsWell) {
                    wellSum++;
                }
                prevIsWell = thisIsWell;
            }
        }
        prevIsWell = false;
        for (int row=0; row<ROWS; row++) {
            thisIsWell = simulatedField[row][COLS-1] == 0 &&
                    simulatedField[row][COLS-2] != 0;
            if (prevIsWell && thisIsWell) {
                wellSum++;
            }
            prevIsWell = thisIsWell;
        }
        features.add((double) wellSum);

        // Original features in the project description
//        // Bias?
//        features.add(1);
        // Column Heights
        for (int i: simulatedTop) {
            features.add((double) i);
        }
        // Difference between adjacent column height
        for (int i=0; i<simulatedTop.length-1; i++)
            features.add((double) Math.abs(simulatedTop[0] - simulatedTop[1]));
        // Maximum column height
        int max = 0;
        for (int i: simulatedTop)
            max = Math.max(i, max);
        features.add((double) max);
//        // Holes
//        int holes = 0;
//        for (int col=0; col<COLS; col++) {
//            for (int row=simulatedTop[col]; row>=0; row--) {
//                if (simulatedField[row][col] == 0) {
//                    holes++;
//                }
//            }
//        }
//        features.add(holes);
//        // Rows cleared
//        features.add(rowsCleared);

        // Convert field to binary input
        int[][] field = super.getField();
        for (int i=0; i<ROWS; i++) {
            for (int j=0; j<COLS; j++) {
                features.add(field[i][j] == 0 ? 0.0 : 1.0);
            }
        }

        // Convert next piece to one-hot input
        int nextPiece = super.getNextPiece();
        for (int i=0; i<N_PIECES; i++) {
            features.add(i == nextPiece ? 1.0 : 0.0);
        }

        return nn.activate(features).get(0);
    }

    public boolean setOutputs(List<Double> outputs) {
        int bestIndex = 0;
        double bestActivation = 0;
        for (int i=0; i<outputs.size(); i++) {
            if (outputs.get(i) > bestActivation) {
                bestActivation = outputs.get(i);
                bestIndex = i;
            }
        }
        if (bestIndex >= legalMoves[nextPiece].length) {
            super.lost = true;
        } else {
            super.makeMove(bestIndex);
        }
        return super.hasLost();
    }

    private int getSlot(List<Double> outputs, int orient) {
        double max = outputs.get(4);
        int slot = 0;
        for (int i=5; i<outputs.size(); i++) {
            if (outputs.get(i) > max) {
                max = outputs.get(i);
                slot = i-4;
            }
        }

        return slot < (COLS - pWidth[super.nextPiece][orient] + 1) ? slot : -1;
    }

    // 0 - box, 1 - line, 2 - L, 3 - L rev, 4 - T, 5 & 6 - Trash
    private int getOrient(List<Double> outputs) {
        int nextPiece = super.getNextPiece();
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

    public double getFitness() {
        return percentageFilled() + super.getRowsCleared();
//        return super.getTurnNumber();
    }

    private double percentageFilled() {
        double noFilled = 0.0;
        // Count no of tiles filled
        // Exclude the top row not visible in-game
        int[][] field = super.getField();
        for (int i=0; i<ROWS-1; i++) {
            for (int j=0; j<COLS; j++) {
                noFilled += field[i][j] > 0 ? 1 : 0;
            }
        }

        return noFilled / ((ROWS-1) * COLS);
    }

    public void makeBestMove() {
        super.makeMove(getBestMove());
    }

    public int getBestMove() {
        return IntStream.range(0, legalMoves().length).parallel().boxed()
                .max(Comparator.comparing(this::evaluateHeuristic))
                .orElse(0);
    }
}
