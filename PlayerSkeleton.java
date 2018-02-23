import java.util.Scanner;

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
		if (!HEADLESS)
			new TFrame(s);
		while(!s.hasLost()) {
			s.makeMove(pickMove(s,s.legalMoves()));
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
