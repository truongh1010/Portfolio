# WumpusWorldAgent
# An agent designed to perform in the wumpus world environment.
# Hao Truong

class WumpusWorldAgent:

    def __init__(self, kb, time=0):
        """
        Initialization
        """
        self.kb = kb
        self.time = time

    def turn_left(self, world):
        """
        Inform the world that the agent has turned left
        """
        print("turning left")
        world.turned_left()

    def turn_right(self, world):
        """
        Inform the world that the agent has turned right
        """
        print("turning right")
        world.turned_right()

    def move_forward(self, world):
        """
        Inform the world that the agent has moved forward 
        """
        print("moving forward")
        world.moved_forward()

    def shoot(self, world):
        """
        Inform the world that the agent has shot
        """
        print("shooting")
        world.shot()

    def grab(self, world):
        """
        Inform the world that the agent has grabbed
        """
        print("grabbing an item")
        world.grabbed()

    def climb(self, world):
        """
        Inform the world that the agent has climbed
        """
        print("climbing out")
        world.climbed()

    def make_percept_sentence(self, percept):
        pass

    def make_action_query(self):
        pass

    def make_action_sentence(self, action):
        pass

    def action(self, percept):
        """
        Determine action based on the percepts and updates KB
        """
        self.kb.tell(self.make_percept_sentence(percept))
        action = self.kb.ask(self.make_action_query())
        self.kb.tell(self.make_action_sentence(action))
        self.time += 1
        return action
