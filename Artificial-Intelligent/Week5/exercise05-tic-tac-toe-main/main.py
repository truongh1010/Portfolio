# Main
# A demonstration of the MinimaxTicTacToeAgent.
# Hao Truong

from tic_tac_toe_game import TicTacToeGame
from tic_tac_toe_board_renderer import TicTacToeBoardRenderer
from human_tic_tac_toe_agent import HumanTicTacToeAgent
from minimax_tic_tac_toe_agent import MinimaxTicTacToeAgent

# Initialize board renderer
renderer = TicTacToeBoardRenderer()

# Create a new Tic-Tac-Toe game
game = TicTacToeGame((None, None, None, None, None, None, None, None, None), renderer)

# Set up agents
human_agent = HumanTicTacToeAgent(game, TicTacToeGame.P1_SYMBOL)  # Human plays X
minimax_agent = MinimaxTicTacToeAgent(game, TicTacToeGame.P2_SYMBOL)  # Minimax AI plays O

# Display game instructions
print(f"Let's play Tic-Tac-Toe! You are {human_agent.symbol} (X) and AI is {minimax_agent.symbol} (O).\n")
print("0 | 1 | 2\n3 | 4 | 5\n6 | 7 | 8\n")

# Game loop
while game.is_not_over():
    print("----------------------------------------------")

    # Human player's turn
    human_move = human_agent.action(game.state)
    game.state = game.result(game.state, human_move)
    print(game)

    # Check if human won
    if game.is_win(game.state, human_agent.symbol):
        print("You win!")
        break

    # AI player's turn
    ai_move = minimax_agent.action(game.state)

    # Handle case where AI has no valid moves
    if ai_move is None:
        print("No valid moves left. The game is a draw!")
        break

    game.state = game.result(game.state, ai_move)
    print(game)

    # Check if AI won
    if game.is_win(game.state, minimax_agent.symbol):
        print("AI wins!")
        break

print("Game Over")