"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return aggressive_score(game, player)


def very_aggressive_score(game, player):
    return difference_in_moves(game, player, my_move_weight=3.0)


def aggressive_score(game, player):
    return difference_in_moves(game, player, my_move_weight=2.0)


def defensive_score(game, player):
    return difference_in_moves(game, player, their_move_weight=2.0)


def defensive_to_aggressive(game, player):
    # sliding scale of 0.5 to 2.0 based on the progress of the game
    spaces = game.width * game.height
    progress = float(game.move_count) / spaces
    base_my_move_weight = 0.5
    my_move_weight = base_my_move_weight + progress * 1.5
    return difference_in_moves(game, player, my_move_weight=my_move_weight)


def aggressive_to_defensive(game, player):
    # sliding scale of 2.0 to 0.5 based on the progress of the game
    spaces = game.width * game.height
    progress = float(game.move_count) / spaces
    base_my_move_weight = 2.0
    my_move_weight = base_my_move_weight - progress * 1.5
    return difference_in_moves(game, player, my_move_weight=my_move_weight)


def difference_in_moves(game, player, my_move_weight=1.0, their_move_weight=1.0):
    utility = game.utility(player)
    in_terminal_state = utility is not 0.
    if in_terminal_state:
        return utility
    my_moves = game.get_legal_moves(player)
    opponent_moves = game.get_legal_moves(game.get_opponent(player))
    return my_move_weight * len(my_moves) - their_move_weight * len(opponent_moves)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=20.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        search_method = self.minimax if self.method is 'minimax' else self.alphabeta
        best_move = (-1, -1)
        if not legal_moves:
            return best_move
        # TODO: Opening book

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if not self.iterative:
                _, best_move = search_method(game, self.search_depth)
            # Perform iterative deepening. On timeout, the best move calculated
            # thus far will be returned.
            else:
                current_depth = 1
                while True:
                    _, best_move = search_method(game, current_depth)
                    current_depth += 1

            return best_move

        except Timeout:
            return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        # assume player we care about is the maximizing player
        player = game.active_player if maximizing_player else game.get_opponent(game.active_player)
        legal_moves = game.get_legal_moves()
        in_terminal_state = not legal_moves

        # Base cases
        if depth is 0 or in_terminal_state:
            return self.score(game, player), (-1, -1)

        # Apply minimax decision to DFS of child scores
        child_scores, _ = zip(*[
            self.minimax(game.forecast_move(move), depth - 1, not maximizing_player)
            for move in legal_moves
        ])
        child_scores_and_moves = list(zip(child_scores, legal_moves))
        minimax_decision = max if maximizing_player else min
        return minimax_decision(child_scores_and_moves, key=self._get_score)

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        # assume player we care about is the maximizing player
        player = game.active_player if maximizing_player else game.get_opponent(game.active_player)
        legal_moves = game.get_legal_moves()
        in_terminal_state = not legal_moves
        best_move = (-1, -1)

        # Base cases
        if depth is 0 or in_terminal_state:
            return self.score(game, player), (-1, -1)

        overall_score = float('-inf') if maximizing_player else float('inf')
        for move in legal_moves:
            score, _ = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, not maximizing_player)
            if maximizing_player:
                if score > overall_score:
                    overall_score = score
                    best_move = move
                alpha = max(alpha, score)
            else:
                if score < overall_score:
                    overall_score = score
                    best_move = move
                beta = min(beta, score)
            if beta <= alpha:
                break

        return overall_score, best_move

    @staticmethod
    def _get_score(search_method_result):
        score, _ = search_method_result
        return score
