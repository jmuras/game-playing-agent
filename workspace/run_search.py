import logging

from run_match import play_matches, TEST_AGENTS, NUM_PROCS, NUM_ROUNDS, TIME_LIMIT
from argparse import Namespace
from isolation import Agent
from my_standard_player import AlphaBetaPlayer

logger = logging.getLogger(__name__)

args = Namespace()
setattr(args, 'debug', False)
setattr(args, 'fair_matches', True)
setattr(args, 'processes', NUM_PROCS)
setattr(args, 'rounds', NUM_ROUNDS)
setattr(args, 'time_limit', TIME_LIMIT)

if __name__ == "__main__":

    test_agent = Agent(AlphaBetaPlayer, "Alpha Beta Agent")
    custom_agent = TEST_AGENTS["SELF"]
    wins, num_games = play_matches(custom_agent, test_agent, args)

    logger.info("Your agent won {:.1f}% of matches against {}".format(
       100. * wins / num_games, test_agent.name))
    print("Your agent won {:.1f}% of matches against {}".format(
       100. * wins / num_games, test_agent.name))
    print()