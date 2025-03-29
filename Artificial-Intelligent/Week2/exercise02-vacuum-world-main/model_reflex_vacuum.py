# ModelReflexVacuum: A robot vacuum cleaner modeled as a model-based reflex agent.
# Your implementation should pass the tests in test_model_reflex_vacuum.py.
# Hao Truong


class ModelReflexVacuum:

    """
    Initialization
    """

    def __init__(self, state, transition_model, sensor_model):
        """
        ModelReflexVaccuum has state, transition_model, sensor_model, and most_recent_action
        """
        self.state = state
        self.transition_model = transition_model
        self.sensor_model = sensor_model
        self.most_recent_action = None

    """
    Sensors
    """

    def sense_dirt(self):
        """
        ModelReflexVaccuum detects whether the current location has dirt
        """
        return self.sensor_model.sense_dirt()
    
    def sense_location_id(self):
        """
        ModelReflexVaccuum determines the current location
        """
        return self.sensor_model.sense_location_id()
    
    """
    Actuators
    """

    def suck(self):
        """
        ModelReflexVaccuum sucks dirt from current location transition 
        model `apply_suction` method
        """
        self.transition_model.apply_suction()
    
    def move_left(self):
        """
        ModelReflexVaccuum moves left from current location
        """
        self.transition_model.move_left()

    def move_right(self):
        """
        ModelReflexVaccuum moves right from current location
        """
        self.transition_model.move_right()

    """
    Update State
    """
    def update_state(self):
        """
        ModelReflexVaccuum tracks changes in the environment over time
        """
        if self.most_recent_action is not None:
            self.most_recent_action()

    """
    Actions
    """

    def action(self):
        """
        ModelReflexVaccuum makes a decision based on sensors, actuators, and environment state
        """

        # Update state
        self.update_state()
        
        # Sense percepts (dirt and location) from sensor model
        is_dirty = self.sense_dirt()
        current_location = self.sense_location_id()

        # Match rules/decide action
        if is_dirty:
            self.most_recent_action = self.suck
        elif current_location == 'A':
            self.most_recent_action = self.move_right
        elif current_location == 'B':
            self.most_recent_action = self.move_left

        return self.most_recent_action
