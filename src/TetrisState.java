import java.util.ArrayList;
import java.util.List;

public class TetrisState extends State {
    public List<Double> getInputs() {
        List<Double> inputs = new ArrayList<>();

        // Convert field to binary input
        int[][] field = super.getField();
        for (int i=0; i<ROWS; i++) {
            for (int j=0; j<COLS; j++) {
                inputs.add(field[i][j] == 0 ? 0.0 : 1.0);
            }
        }

        // Convert next piece to one-hot input
        int nextPiece = super.getNextPiece();
        for (int i=0; i<N_PIECES; i++) {
            inputs.add(i == nextPiece ? 1.0 : 0.0);
        }

        // Feature Information Start!

        // column heights
        int maxColumnHeight = 0;
        for (int i=0; i<COLS; i++) {
            for (int j=ROWS-1; j>=0; j--) {
                if ((field[j][i]!=0)) {
                    if (maxColumnHeight < j+1)
                        maxColumnHeight = j+1;
                    inputs.add(j + 1.0);
                    break;
                }
                else if (j == 0)
                    inputs.add((double)j);
            }
        }

        // absolute differences between adjacent column heights
        for (int i=0; i<COLS - 1; i++) {
            inputs.add(Math.abs(inputs.get(inputs.size()-COLS) - inputs.get(inputs.size()+1-COLS)));
        }

        // 1 max column height
        inputs.add((double)maxColumnHeight);

        // 1 number of holes in the wall
        int holes = 0;
        boolean underThing = false;
        for (int i=0; i<COLS; i++) {
            for (int j=ROWS-1; j>=0; j--) {
                if (field[j][i]!=0 && !underThing) {
                    underThing = true;
                }
                else if (field[j][i]==0 && underThing) {
                    holes++;
                }
            }
            underThing = false;
        }
        inputs.add((double)holes);

        return inputs;
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
//        return percentageFilled() + super.getRowsCleared();
        return super.getTurnNumber();
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
}
