# SimpleReflexVacuum: A robot vacuum cleaner modeled as a simple reflex agent.
# Your implementation should pass the tests in test_simple_reflex_vacuum.py.
# Hao Truong


class SimpleReflexVacuum:

    def suck(self):
        pass

    def move_left(self):
        pass

    def move_right(self):
        pass

    def action(self, location_id, dirt):
        if dirt == 'Dirt':
            return self.suck
        elif location_id == 'A' and dirt is None:
            return self.move_right
        elif location_id == 'B' and dirt is None:
            return self.move_left
