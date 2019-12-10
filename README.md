# NEAT-Tetris
Evolving an AI to play Tetris Using Neuroevolution of Augmenting Topologies

In this project, we attempt to use the [the NeuroEvolution of Augmenting Topologies (NEAT)](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) as an algorithm for learning Tetris. By using NEAT over a typical neural network, it may be able to learn better than a typical neural network, with the ability to learn it's structure, adapting not only the weights but which nodes the links connect to and providing the ability to jump between layers, reducing the impact of vanishing gradient.

## Result

Our final score is 830000 lines cleared. The results are unexpectedly good because it was able to evolve so quickly in merely 12 generations. The average time taken to complete the 830000 with the tetris GUI is approximately 118 minutes.

## Files	
### State
This is the tetris simulation.  It keeps track of the state and allows you to make moves.  The board state is stored in field (a double array of integers) and is accessed by getField(). Zeros denote an empty square.  Other values denote the turn on which that square was placed.  NextPiece (accessed by getNextPiece) contains the ID (0-6) of the piece you are about to play.

Moves are defined by two numbers: the SLOT, the leftmost column of the piece and the ORIENT, the orientation of the piece.  Legalmoves gives an nx2 int array containing the n legal moves.  A move can be made by specifying the two parameters as either 2 ints, an int array of length 2, or a single int specifying the row in the legalMoves array corresponding to the appropriate move.

It also keeps track of the number of lines cleared - accessed by getRowsCleared().

draw() draws the board.

drawNext() draws the next piece above the board

clearNext() clears the drawing of the next piece so it can be drawn in a different slot/orientation

### TFrame 
This extends JFrame and is instantiated to draw a state.

It can save the current drawing to a .png file.

The main function allows you to play a game manually using the arrow keys.

### TLabel
This is a drawing library.

### PlayerSkeleton
The actual player implementation.

The main function performs the training across generations, showcasing the best results every 5 generations.
