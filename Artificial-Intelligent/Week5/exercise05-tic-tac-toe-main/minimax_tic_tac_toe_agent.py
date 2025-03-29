# MinimaxTicTacToeAgent
# A game-playing tic tac toe agent that uses the minimax algorithm to produce
# a rational action.
# Hao Truong


class MinimaxTicTacToeAgent:

    def __init__(self, game, symbol):
        self.game = game
        self.symbol = symbol

    def action(self, state):
        if self.game is None:
            return None
        
        _, move = self.minimax(self.game, state)
        return move

    def minimax(self, game, state):
        """
        this performs Minimax algorithm to find the best move assuming the oponent plays optimally
        """
        if game.is_terminal(state):
            return game.utility(state, self.symbol), None
        
        player = game.to_move(state)

        return self.max_value(game, state) if player == self.symbol else self.min_value(game, state)

    def max_value(self, game, state):
        """
        when it's the agent's turn (maximizing player), the agent explors all possible
        moves and return the one with the highest utility score
        """
        if game.is_terminal(state):
            return game.utility(state, self.symbol), None
        
        v = -float('inf')
        move = None

        for a in game.actions(state):
            v2, a2 = self.min_value(game, game.result(state, a))

            if v2 > v:
                v, move = v2, a

        return v, move

    def min_value(self, game, state):
        """
        when it's the oponent's turn (minimizing player), the agent explors all possible
        moves and return the one with the lowest utility score
        """
        if game.is_terminal(state):
            return game.utility(state, self.symbol), None
        
        v = float('inf')
        move = None

        for a in game.actions(state):
            v2, a2 = self.max_value(game, game.result(state, a))

            if v2 < v:
                v, move = v2, a

        return v, move
