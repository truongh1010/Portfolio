# KnowledgeBase
# A knowledge base for a knowledge-based agent.
# Hao Truong

from wumpus_world_agent import WumpusWorldAgent

class KnowledgeBase:

    def __init__(self):
        """
        Initialization
        """
        self.facts = set()

    def tell(self, fact):
        """
        Add a fact into KB
        """
        self.facts.add(fact)

    def ask(self,query):
        """
        Retrieve an action from KB. For now, stub the implementation 
        by return the actual climb function from WumpusWorldAgent.
        """
        return WumpusWorldAgent.climb
