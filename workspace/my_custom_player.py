import random
import logging
import time

from sample_players import DataPlayer
from isolation.isolation import _SIZE

class CustomPlayer(DataPlayer):

    """ Implement your own agent to play knight's Isolation

    The get_action() method is the only required method for this project.
    You can modify the interface for get_action by adding named parameters
    with default values, but the function MUST remain compatible with the
    default interface.

    **********************************************************************
    NOTES:
    - The test cases will NOT be run on a machine with GPU access, nor be
      suitable for using any other machine learning techniques.

    - You can pass state forward to your agent on the next turn by assigning
      any pickleable object to the self.context attribute.
    **********************************************************************
    """
    def get_action(self, state):
        """ Employ an adversarial search technique to choose an action
        available in the current state calls self.queue.put(ACTION) at least

        This method must call self.queue.put(ACTION) at least once, and may
        call it as many times as you want; the caller will be responsible
        for cutting off the function after the search time limit has expired.

        See RandomPlayer and GreedyPlayer in sample_players for more examples.

        **********************************************************************
        NOTE: 
        - The caller is responsible for cutting off search, so calling
          get_action() from your own code will create an infinite loop!
          Refer to (and use!) the Isolation.play() function to run games.
        **********************************************************************
        """
        # TODO: Replace the example implementation below with your own search
        #       method by combining techniques from lecture - minimax search
        #       with alpha-beta pruning and iterative deepening
        #
        # EXAMPLE: choose a random move without any search--this function MUST
        #          call self.queue.put(ACTION) at least once before time expires
        #          (the timer is automatically managed for you)

        # if self.context is not None:
        #     self.logResults(self.context)

        if state.ply_count < 2:
            self.queue.put(random.choice(state.actions()))
        else:
            for depth in range(3, _SIZE):
                action = self.minimax(state, depth=depth)
                self.queue.put(action)
                # self.context = (state.ply_count, depth)

    def minimax(self, state, depth):

        alpha = float("-inf")
        beta = float("inf")

        def min_value(state, alpha, beta, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("inf")
            for action in state.actions():
                value = min(value, max_value(state.result(action), alpha, beta, depth - 1))
                if value <= alpha:
                    return value
                beta = min(beta, value)
            return value

        def max_value(state, alpha, beta, depth):
            if state.terminal_test(): return state.utility(self.player_id)
            if depth <= 0: return self.score(state)
            value = float("-inf")
            for action in state.actions():
                value = max(value, min_value(state.result(action), alpha, beta, depth - 1))
                if value >= beta:
                    return value
                alpha = max(alpha, value)
            return value

        return max(state.actions(), key=lambda x: min_value(state.result(x), alpha, beta, depth - 1))

    def score(self, state):

        own_loc = state.locs[self.player_id]
        opp_loc = state.locs[1 - self.player_id]
        own_liberties = state.liberties(own_loc)
        opp_liberties = state.liberties(opp_loc)

        score = (2 * len(own_liberties) * state.ply_count / _SIZE) - len(opp_liberties)

        # count = 0
        # for own_liberty in own_liberties:
        #     for opp_liberty in opp_liberties:
        #         if own_liberty == opp_liberty:
        #             count = count + 1
        # score = len(own_liberties) - 2 * len(opp_liberties) + count

        # score = len(own_liberties) - len(opp_liberties)

        return score

    # def logResults(self, context):
    #     f = open("my_player.txt", "a")
    #     f.write(str(context) + "\n")
    #     f.close()