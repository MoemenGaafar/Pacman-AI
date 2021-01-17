# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()  # Current (x,y) coordinates
        newFood = successorGameState.getFood()  # Boolean grid of all food
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        newGhostPos = successorGameState.getGhostPositions()
        newScore = successorGameState.getScore()

        "*** YOUR CODE HERE ***"
        # Calculate total reciprocal distance to all ghosts
        ghostDistance = 0
        for ghostPos in newGhostPos:
            if ghostPos == newPos:
                ghostDistance += float('inf')
            else:
                ghostDistance += 1 / manhattanDistance(ghostPos, newPos)

        # Calculate total reciprocal distance to all food capsules.
        rows = range(len(list(newFood)))
        cols = range(len(list(newFood[0])))
        foodDistance = 0
        for r in rows:
            for c in cols:
                if newFood[r][c]:
                    if (r, c) == newPos:
                        foodDistance += 0.1
                    else:
                        foodDistance += 0.1 * 1 / manhattanDistance((r, c), newPos)

        # Our evaluation of the action is directly proportional to the new score and the reciprocal of food distances,
        # and inversely proportional to the ghost distances
        evaluation = newScore + foodDistance - ghostDistance

        return evaluation


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def maxValue(self, gameState, currentDepth):
        """
        This function implements the functionality of a max node in a minimax tree.
        """
        # Check for leaves
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        # Check if maximum depth is reached
        if currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        maxi = float('-inf')
        # Return the action with the maximum evaluation
        for action in gameState.getLegalActions():
            maxi = max(maxi, self.minValue(gameState.generateSuccessor(0, action), currentDepth))
        return maxi

    def minValue(self, gameState, currentDepth, iGhost=1):
        """
        This function implements the functionality of a min node in a minimax tree. It calls itself once for every
        ghost, then calls the max function when it is done.
        """
        # Check for leaves
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        # Return the action with the minimum evaluation
        mini = float('inf')
        ghostActions = gameState.getLegalActions(iGhost)
        if iGhost == gameState.getNumAgents() - 1:
            # If this is the last ghost to take action, call Pacman's max node
            for gAction in ghostActions:
                nextState = gameState.generateSuccessor(iGhost, gAction)
                mini = min(mini, self.maxValue(nextState, currentDepth + 1))
        else:
            # Otherwise, call the next ghost's min node
            for gAction in ghostActions:
                nextState = gameState.generateSuccessor(iGhost, gAction)
                mini = min(mini, self.minValue(nextState, currentDepth, iGhost + 1))
        return mini

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return max(((self.minValue(gameState.generateSuccessor(0, action), 0), action)
                    for action in gameState.getLegalActions()), key=lambda entry: entry[0])[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def maxValue(self, gameState, currentDepth, alpha, beta):
        """
        This function implements the functionality of a max node in a minimax tree.
        """
        # Check for leaves
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        # Check if maximum depth is reached
        if currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        # Return the action with the maximum evaluation
        maxi = float('-inf')
        for action in gameState.getLegalActions():
            maxi = max(maxi, self.minValue(gameState.generateSuccessor(0, action), currentDepth, alpha, beta))
            # Pruning
            if maxi > beta:
                return maxi
            alpha = max(alpha, maxi)
        return maxi

    def minValue(self, gameState, currentDepth, alpha, beta, iGhost=1):
        """
        This function implements the functionality of a min node in a minimax tree. It calls itself once for every
        ghost, then calls the max function when it is done.
        """
        # Check for leaves
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        # Return the action with the minimum evaluation
        mini = float('inf')
        ghostActions = gameState.getLegalActions(iGhost)
        if iGhost == gameState.getNumAgents() - 1:
            # If this is the last ghost to take action, call Pacman's max node
            for gAction in ghostActions:
                nextState = gameState.generateSuccessor(iGhost, gAction)
                mini = min(mini, self.maxValue(nextState, currentDepth + 1, alpha, beta))
                # Pruning
                if mini < alpha:
                    return mini
                beta = min(beta, mini)
        else:
            # Otherwise, call the next ghost's min node
            for gAction in ghostActions:
                nextState = gameState.generateSuccessor(iGhost, gAction)
                mini = min(mini, self.minValue(nextState, currentDepth, alpha, beta, iGhost + 1))
                # Pruning
                if mini < alpha:
                    return mini
                beta = min(beta, mini)
        return mini

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # Initialize the values of alpha and beta
        alpha = float('-inf')
        beta = float('inf')
        actionValue = (float('-inf'),)
        # Generate the first Pacman action using a max node
        for action in gameState.getLegalActions():
            actionValue = max(actionValue, (self.minValue(gameState.generateSuccessor(0, action), 0, alpha, beta),
                                            action), key=lambda entry: entry[0])
            # Pruning
            if actionValue[0] > beta:
                return action
            alpha = max(alpha, actionValue[0])
        return actionValue[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    This function implements the functionality of a max node in a minimax tree.
    """
    def maxValue(self, gameState, currentDepth):
        # Check for leaves
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        # Check if maximum depth is reached
        if currentDepth == self.depth:
            return self.evaluationFunction(gameState)
        # Return the action with the maximum evaluation
        maxi = float('-inf')
        for action in gameState.getLegalActions():
            maxi = max(maxi, self.chanceValue(gameState.generateSuccessor(0, action), currentDepth))
        return maxi

    def chanceValue(self, gameState, currentDepth, iGhost=1):
        """
        This function implements the functionality of a chance node in an expectimax tree. It calls itself once for
        every ghost, then calls the max function when it is done.
        """
        # Check for leaves
        if gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)
        # Generate all possible ghost actions
        ghostActions = gameState.getLegalActions(iGhost)
        # Assuming uniform probabilities, calculate the probability of each action
        chance = 1 / len(ghostActions)
        totalChance = 0
        # Return the average evaluation of all actions
        if iGhost == gameState.getNumAgents() - 1:
            # If this is the last ghost to take action, call Pacman's max node
            for gAction in ghostActions:
                nextState = gameState.generateSuccessor(iGhost, gAction)
                totalChance += chance * self.maxValue(nextState, currentDepth + 1)
        else:
            # Otherwise, call the next ghost's min node
            for gAction in ghostActions:
                nextState = gameState.generateSuccessor(iGhost, gAction)
                totalChance += chance * self.chanceValue(nextState, currentDepth, iGhost + 1)

        return totalChance

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return max(((self.chanceValue(gameState.generateSuccessor(0, action), 0), action)
                    for action in gameState.getLegalActions()), key=lambda entry: entry[0])[1]


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I designed an evaluation function that relies on three main heuristics to determine which action is
    more favorable. The first is the game score. This heuristic in itself includes many other heuristics,
    like the number of food capsules eaten, the number of seconds that have passed and whether Pacman wins or loses.
    The second is the distance between Pacman and food capsules. The aim is to favor states in which Pacman is closer
    to food. But rather than implementing a linear function, it makes more sense that Pacman would favor food
    capsules that are near him than those which are far away. That is why I calculate the sum of reciprocals of food
    distances and aim to maximize this sum. The third is the distance between Pacman and ghosts. The aim is to favor
    states in which Pacman is further away from ghosts. But rather than implementing a linear function, it makes more
    sense that Pacman would be more afraid of ghosts that are near him than those which are far away. That is why I
    calculate the sum of reciprocals of ghost distances and aim to minimize this sum.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()  # Current (x,y) coordinates
    newFood = currentGameState.getFood()  # Boolean grid of all food
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newGhostPos = currentGameState.getGhostPositions()
    newScore = currentGameState.getScore()

    # Calculate total reciprocal distance to all ghosts
    ghostDistance = 0
    for ghostPos in newGhostPos:
        if ghostPos == newPos:
            ghostDistance += float('inf')
        else:
            ghostDistance += 1 / manhattanDistance(ghostPos, newPos)

    # Calculate total reciprocal distance to all food capsules
    rows = range(len(list(newFood)))
    cols = range(len(list(newFood[0])))
    foodDistance = 0
    for r in rows:
        for c in cols:
            if newFood[r][c]:
                if (r, c) == newPos:
                    foodDistance += 1
                else:
                    foodDistance += 1 / manhattanDistance((r, c), newPos)

    # Our evaluation of the action is directly proportional to the new score and the reciprocal of food distances,
    # and inversely proportional to the ghost distances
    evaluation = newScore + foodDistance - ghostDistance

    return evaluation


# Abbreviation
better = betterEvaluationFunction
