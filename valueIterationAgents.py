# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
          temp = util.Counter()
          for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
              self.values[state] = self.mdp.getReward(state, 'exit', '')
            else:
              actions = self.mdp.getPossibleActions(state)
              temp[state] = max([self.computeQValueFromValues(state, action) for action in actions])
          self.values = temp

        

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        value = 0
        for ts in transitionStatesAndProbs:
          value += self.mdp.getReward(state, action, ts[0]) + self.discount*(self.values[ts[0]]*ts[1])

        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        stateAction = util.Counter()
        for actions in self.mdp.getPossibleActions(state):
          stateAction[actions] = self.computeQValueFromValues(state,actions)
        return stateAction.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
          for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
              self.values[state] = self.mdp.getReward(state, 'exit', '')
            else:
              actions = self.mdp.getPossibleActions(state)
              self.values[state] = max([self.computeQValueFromValues(state, action) for action in actions])

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessors = dict()
        s = self.mdp.getStates()

        for state in s:
          actions = self.mdp.getPossibleActions(state)
          for action in actions:
            nextstates = self.mdp.getTransitionStatesAndProbs(state,action)
            for newstate in nextstates:
              #print("newstate",newstate)
              try:
                #print("try add")
                #print(str(newstate[0]),"to",state)
                predecessors[str(newstate[0])].add(state)
                #print("result:",str(newstate[0]),":",predecessors[str(newstate[0])])
                #print(type(predecessors[str(newstate[0])]))
              except KeyError:
                predecessors[str(newstate[0])] = {state}
                #print("result:",str(newstate[0]),":",predecessors[str(newstate[0])])
                #print(type(predecessors[str(newstate[0])]))
        pq = util.PriorityQueue()
        for state in s:
          actions = self.mdp.getPossibleActions(state)
          if len(actions) != 0:
            diff = abs(max([self.computeQValueFromValues(state, action) for action in actions]) - self.values[state])
            pq.update(state, -diff)
        for i in range(self.iterations):
          print(pq.count)
          if not pq.isEmpty():
            state = pq.pop()
            if self.mdp.isTerminal(state):
              self.values[state] = self.mdp.getReward(state, 'exit', '')
            else:
              actions = self.mdp.getPossibleActions(state)
              self.values[state] = max([self.computeQValueFromValues(state, action) for action in actions])

            for p in predecessors[str(state)]:
              actions = self.mdp.getPossibleActions(p)
              diff = abs(max([self.computeQValueFromValues(p, action) for action in actions])-self.values[p])
              if diff > self.theta:
                pq.update(p, -diff)





