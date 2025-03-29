# WumpusWorld
# A simulated representation of a real Wumpus World, aligned with the specified
# characteristics in the AIMA text.
# Note: This is not a state model. It _is_ the real world / environment within
# which the agent operates. Think of it as actual, physical, reality.
# Note 2: This simulation will not include the modeling of time, for the sake of
# simplicity. This only affects the 'Bump' and 'Scream' percepts. In the case of
# 'Bump', we assume that when an agent is in a room facing a wall, it should receive
# the 'Bump' percept. For 'Scream', when the wumpus is killed we let the scream
# linger throughout the cave indefinitely.
# Hao Trung

class WumpusWorld:

    EXIT_LOCATION = (1, 1)
    GRID_SIZE = 4

    def __init__(self, agent_location = (1,1), agent_direction = 'East',
                       agent_alive = True, wumpus_alive = None,
                       wumpus_location = None, gold_location = None,
                       pit_locations = []):
        self.agent_location = agent_location
        self.agent_direction = agent_direction
        self.agent_alive = agent_alive
        self.wumpus_alive = wumpus_alive
        self.wumpus_location = wumpus_location
        self.gold_location = gold_location
        self.pit_locations = pit_locations

    def percept(self, location):
        """
        The current five-element percept in the `location`. Returns a tuple in the
        form of ('Stench', 'Breeze', 'Glitter', 'Bump', 'Scream'). Any of the elements
        within the returned percept tuple can be None.
        """
        if location is None:
            return (None, None, None, None, None)
        
        glitter = 'Glitter' if location == self.gold_location else None
        stench = 'Stench' if location == self.wumpus_location or self.adjacent(location, self.wumpus_location) else None
        breeze = 'Breeze' if any(self.adjacent(location, pit) for pit in self.pit_locations) else None
        bump = 'Bump' if location == self.agent_location and self.agent_bumped_wall() else None
        scream = 'Scream' if self.wumpus_alive == False else None

        return (stench, breeze, glitter, bump, scream)



    """
    Physical side effects of agent actions
    """

    def turned_left(self):
        """
        Turn the agent counter-clockwise, to the left, resulting in a new
        `agent_direction`.
        """
        if self.agent_direction == 'East':
            self.agent_direction = 'North'
        elif self.agent_direction == 'North':
            self.agent_direction = 'West'
        elif self.agent_direction == 'West':
            self.agent_direction = 'South'
        elif self.agent_direction == 'South':
            self.agent_direction = 'East'

    def turned_right(self):
        """
        Turn the agent clockwise, to the right, resulting in a new `agent_direction`.
        """
        if self.agent_direction == 'East':
            self.agent_direction = 'South'
        elif self.agent_direction == 'South':
            self.agent_direction = 'West'
        elif self.agent_direction == 'West':
            self.agent_direction = 'North'
        elif self.agent_direction == 'North':
            self.agent_direction = 'East'

    def moved_forward(self):
        """
        Attempt to move forward. When successful, update the agent location.
        Moving into a pit location kills the agent.
        Moving into a living wumpus location kills the agent.
        """
        if not self.agent_bumped_wall():
            agent_location_x, agent_location_y = self.agent_location

            if self.agent_direction == 'East':
                self.agent_location = (agent_location_x + 1, agent_location_y)
            elif self.agent_direction == 'West':
                self.agent_location = (agent_location_x - 1, agent_location_y)
            elif self.agent_direction == 'North':
                self.agent_location = (agent_location_x, agent_location_y + 1)
            elif self.agent_direction == 'South':
                self.agent_location = (agent_location_x, agent_location_y - 1)

            if self.agent_location in self.pit_locations:
                self.agent_alive = False
            
            if self.agent_location == self.wumpus_location and self.wumpus_alive:
                self.agent_alive = False

    def grabbed(self):
        """
        Attempt to grab the gold. Successful when executed in `gold_location`, in
        which case the gold location should be set to None.
        """
        if self.agent_location == self.gold_location:
            self.gold_location = None

    def climbed(self):
        """
        Attempt to climb out of the cave. Successful when executed in location
        (1, 1), in which case the agent location should be set to None.
        """
        if self.agent_location == self.EXIT_LOCATION:
            self.agent_location = None

    def shot(self):
        """
        Shoot the arrow. If the arrow strikes the wumpus, then the wumpus should
        no longer be alive.
        """
        if self.agent_direction == 'East' and self.wumpus_east_of_agent():
            self.wumpus_alive = False
        elif self.agent_direction == 'North' and self.wumpus_north_of_agent():
            self.wumpus_alive = False
        elif self.agent_direction == 'West' and self.wumpus_west_of_agent():
            self.wumpus_alive = False
        elif self.agent_direction == 'South' and self.wumpus_south_of_agent():
            self.wumpus_alive = False

    """
    Helper methods
    """

    def adjacent(self, location, target):
        """
        Is `location` immediately north, south, east or west of `target`?
        """
        if location is None or target is None:
            return False
        
        location_x,location_y = location
        target_x, target_y = target

        return (location_x == target_x and abs(location_y - target_y) == 1) or (location_y == target_y and abs(location_x - target_x) == 1)

    def agent_can_move_east(self):
        """
        Can the agent move east?
        """
        agent_location_x, agent_location_y = self.agent_location

        # the agent can move East if it is not at the rightmost position
        if agent_location_x < 4:
            return True
        return False

    def agent_can_move_west(self):
        """
        Can the agent move west?
        """
        agent_location_x, agent_location_y = self.agent_location

        # the agent can move West if it is not at the leftmost position
        if agent_location_x > 1:
            return True
        return False

    def agent_can_move_north(self):
        """
        Can the agent move north?
        """
        agent_location_x, agent_location_y = self.agent_location

        # the agent can move North if it is not at the top-most position
        if agent_location_y < 4:
            return True
        return False

    def agent_can_move_south(self):
        """
        Can the agent move south?
        """
        agent_location_x, agent_location_y = self.agent_location

        # the agent can move South if it is not at the bottom-most position
        if agent_location_y > 1:
            return True
        return False

    def agent_bumped_wall(self):
        """
        Did the agent bump into a wall? (Or, is the agent facing a wall?)
        """
        agent_location_x, agent_location_y = self.agent_location

        if agent_location_x == 1 and self.agent_direction == 'West':
            return True
        if agent_location_y == 1 and self.agent_direction == 'South':
            return True
        if agent_location_x == 4  and self.agent_direction == 'East':
            return True
        if agent_location_y == 4 and self.agent_direction == 'North':
            return True
        return False

    def wumpus_east_of_agent(self):
        """
        Is the wumpus somewhere to the east of the agent?
        """
        agent_location_x, agent_location_y = self.agent_location
        wumpus_location_x, wumpus_location_y = self.wumpus_location

        if wumpus_location_x > agent_location_x and wumpus_location_y == agent_location_y:
            return True
        return False

    def wumpus_west_of_agent(self):
        """
        Is the wumpus somewhere to the west of the agent?
        """
        agent_location_x, agent_location_y = self.agent_location
        wumpus_location_x, wumpus_location_y = self.wumpus_location

        if wumpus_location_x < agent_location_x and wumpus_location_y == agent_location_y:
            return True
        return False

    def wumpus_north_of_agent(self):
        """
        Is the wumpus somewhere to the north of the agent?
        """
        agent_location_x, agent_location_y = self.agent_location
        wumpus_location_x, wumpus_location_y = self.wumpus_location

        if wumpus_location_y > agent_location_y and wumpus_location_x == agent_location_x:
            return True
        return False

    def wumpus_south_of_agent(self):
        """
        Is the wumpus somewhere to the south of the agent?
        """
        agent_location_x, agent_location_y = self.agent_location
        wumpus_location_x, wumpus_location_y = self.wumpus_location

        if wumpus_location_y < agent_location_y and wumpus_location_x == agent_location_x:
            return True
        return False
