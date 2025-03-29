# Main
# A demonstration of the WumpusWorld and WumpusWorldAgent.
# Hao Truong

from wumpus_world import WumpusWorld
from wumpus_world_agent import WumpusWorldAgent
from knowledge_base import KnowledgeBase

def main():
    world = WumpusWorld(
        agent_location=(1, 1),
        agent_direction='East',
        agent_alive=True,
        wumpus_alive=True,
        wumpus_location=(1, 3),
        gold_location=(2, 3),
        pit_locations=[(3, 1), (3, 3), (4, 3)]
    )

    kb = KnowledgeBase()
    agent = WumpusWorldAgent(kb)

    print("\Starting Wumpus World Simulation...\n")

    for step in range(15):  # Run a limited number of steps
        print(f"Step {step + 1}: Agent at {world.agent_location}, Facing {world.agent_direction}")
        
        percept = world.percept(world.agent_location)
        print(f"Perceived: {percept}")

        action = agent.action(percept)  
        print(f"Action Chosen: {action}")

        if action == "climb":
            print("Agent climbs out and escapes!\n")
            break

        # we have not finished, so far the only action we can take is climb
        print(f"action `{action}`— this is the only action handled!\n")

    print("\nSimulation Ended.\n")

if __name__ == "__main__":
    main()
