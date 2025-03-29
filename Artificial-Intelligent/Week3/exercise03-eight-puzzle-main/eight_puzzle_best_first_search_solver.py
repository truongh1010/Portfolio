# EightPuzzleBestFirstSearchSolver: A problem solver for the eight-puzzle problem
# that can apply best-first search to find a solution node. This class encapsulates
# a best-first search algorithm and an evaluation function. It encapsulates the
# application of the algorithm to the problem, and in the end can produce a
# solution, which is a list of actions.

from queue import PriorityQueue
from eight_puzzle_node import EightPuzzleNode

class EightPuzzleBestFirstSearchSolver:

    def solution(self, problem):
        """
        Return a list of EightPuzzleAgent actuator methods. If the problem
        initial state is the same as the goal state, return an empty list.
        """
        solution_node = self.best_first_search(problem,
            self.cost_so_far_plus_estimated_cost_remaining)
        if solution_node:
            return self.actions_to_reach_solution_node(solution_node)
        else:
            return None

    def best_first_search(self, problem, evaluation_function):
        """
        Return a solution EightPuzzleNode, or None to indicate failure.
        """
        initial_node = EightPuzzleNode(state=problem.initial_state, parent=None, action=None, path_cost=0)
        frontier = PriorityQueue()
        frontier.put((evaluation_function(initial_node), initial_node))
        reached = {problem.initial_state: initial_node}
        while not frontier.empty():
            _, current_node = frontier.get()
            if current_node.state == problem.goal_state:
                return current_node           
            for child_node in self.expand(problem, current_node):
                state = child_node.state
                if state not in reached or child_node.path_cost < reached[state].path_cost:
                    reached[state] = child_node
                    frontier.put((evaluation_function(child_node), child_node))
        return None

    def expand(self, problem, node):
        """
        Return a list of EightPuzzleNodes that are reachable from `node`.
        """
        successors = []
        for action in problem.actions(node.state):
            successor_state = problem.result(node.state, action)
            cost = node.path_cost + problem.action_cost(node.state, action, successor_state)
            successor_node = EightPuzzleNode(
                state=successor_state,
                parent=node,
                action=action,
                path_cost=cost
            )
            successors.append(successor_node)
        return successors

    def cost_so_far_plus_estimated_cost_remaining(self, node):
        """
        The evaluation function, f(n) = g(n) + h(n).
        """
        def heuristic(state):
            """
            Calculate the total Manhattan distance for all tiles.
            """
            distance = 0
            goal_positions = {
                None: (0, 0),
                1: (0, 1), 2: (0, 2),
                3: (1, 0), 4: (1, 1), 5: (1, 2),
                6: (2, 0), 7: (2, 1), 8: (2, 2)
            }
            for i, tile in enumerate(state):
                if tile is not None:
                    x1, y1 = divmod(i, 3)
                    x2, y2 = goal_positions[tile]
                    distance += abs(x1 - x2) + abs(y1 - y2)
            return distance

        return node.path_cost + heuristic(node.state)

    def actions_to_reach_solution_node(self, solution_node):
        """
        Given an EightPuzzleNode goal node, produce a list of in-order actions
        that lead from the initial state to the goal state.
        """
        actions = []
        current_node = solution_node
        while current_node.parent is not None:
            actions.append(current_node.action)
            current_node = current_node.parent
        return list(reversed(actions))
