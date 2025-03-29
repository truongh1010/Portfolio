# EightPuzzleNode: A node in the search graph of the eight puzzle problem.
# As specified by Russell & Norvig, a Node has a state, a parent node, an action,
# and a path cost.

class EightPuzzleNode:

    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __lt__(self, other):
        """
        Define less-than for proper comparison in PriorityQueue.
        Compare based on path_cost or any other criteria as needed.
        """
        return self.path_cost < other.path_cost
