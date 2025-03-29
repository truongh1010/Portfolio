# EightPuzzleAgent: A goal-based agent that emits the actions from a solution to
# an "eight puzzle" problem.
# Your implementation should pass the tests in test_eight_puzzle_agent.py.
# Hao Truong


class EightPuzzleAgent:

    def __init__(self, initial_state, transition_model, actions):
        self.current_state = initial_state
        self.transition_model = transition_model
        self.actions = actions

    def has_actions(self):
        """ Return `True` when `self.actions` is not empty """
        return len(self.actions) > 0

    def action(self):
        """ Return the next action in `self.actions` """
        if self.has_actions():
            return self.actions.pop(0)
        return None

    def move_left(self):
        """
        Print "Left" as a side effect, and update `self.current_state`. Use the
        EightPuzzleTransitionModel stored in `self.transition_model` to obtain
        the new state to assign to `self.current_state`.
        """
        print("Left")
        self.current_state = self.transition_model.move_left(self.current_state)

    def move_right(self):
        """
        Print "Right" as a side effect, and update `self.current_state`. Use the
        EightPuzzleTransitionModel stored in `self.transition_model` to obtain
        the new state to assign to `self.current_state`.
        """
        print("Right")
        self.current_state = self.transition_model.move_right(self.current_state)

    def move_up(self):
        """
        Print "Up" as a side effect, and update `self.current_state`. Use the
        EightPuzzleTransitionModel stored in `self.transition_model` to obtain
        the new state to assign to `self.current_state`.
        """
        print("Up")
        self.current_state = self.transition_model.move_up(self.current_state)

    def move_down(self):
        """
        Print "Down" as a side effect, and update `self.current_state`. Use the
        EightPuzzleTransitionModel stored in `self.transition_model` to obtain
        the new state to assign to `self.current_state`.
        """
        print("Down")
        self.current_state = self.transition_model.move_down(self.current_state)
