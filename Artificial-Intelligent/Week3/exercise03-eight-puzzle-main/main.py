# Main
# Hao Truong
# Demonstrate the use of your EightPuzzleAgent.

from eight_puzzle_agent import EightPuzzleAgent
from eight_puzzle_best_first_search_solver import EightPuzzleBestFirstSearchSolver
from eight_puzzle_transition_model import EightPuzzleTransitionModel
from eight_puzzle_problem import EightPuzzleProblem


def demo_problem_solver(initial_state, goal_state):
    """
    Demonstrate solving an eight-puzzle problem with the provided initial and goal states.
    """
    print(f"Initial State: {initial_state}")
    print(f"Goal State: {goal_state}")

    transition_model = EightPuzzleTransitionModel()
    problem = EightPuzzleProblem(initial_state, goal_state, transition_model)
    solver = EightPuzzleBestFirstSearchSolver()
    
    actions = solver.solution(problem)
    if actions is None:
        print("No solution found!")
        print("-" * 50)
        return

    agent = EightPuzzleAgent(initial_state, transition_model, actions)

    while agent.has_actions():
        action = agent.action()
        action(agent)

    print(agent.current_state)
    print("Problem solved!")
    print("-" * 50)

if __name__ == "__main__":

    # Goal State
    goal_state = (None, 1, 2, 3, 4, 5, 6, 7, 8)

    # Challenging Initial States
    initial_states = [
        (7, 2, 4, 5, None, 6, 8, 3, 1),
        (8, None, 6, 5, 4, 7, 2, 3, 1),
        (8, 6, 7, 2, 5, 4, 3, None, 1),
        (6, 4, 7, 8, 5, None, 3, 2, 1)
    ]

    # Test each initial state
    for initial_state in initial_states:
        demo_problem_solver(initial_state, goal_state)
