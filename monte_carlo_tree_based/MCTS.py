"""Monte Carlo Tree Search for fixed-size feature subsets."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

try:  # package execution
    from .feature_selection import FeatureSelectionState
except ImportError:  # direct execution from this directory
    from feature_selection import FeatureSelectionState


class MCTSNode:
    """One partial feature subset in the search tree."""

    def __init__(
        self,
        game_state: FeatureSelectionState,
        parent: Optional["MCTSNode"] = None,
        move: Optional[int] = None,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.game_state = game_state
        self.parent = parent
        self.move = move # Stores the move that produced this node from parent
        self.children: list[MCTSNode] = []
        self.total_reward = 0.0
        self.visits = 0
        self.untried_moves = game_state.get_legal_moves()
        (rng or random).shuffle(self.untried_moves)

    @property
    def mean_reward(self) -> float: # Calculates average reward for rollout going through this node
        return self.total_reward / self.visits if self.visits else 0.0

    def is_terminal(self) -> bool: # Checks wether state is terminal using feature_selection class's function
        return self.game_state.is_terminal()

    def can_expand(self, widening_constant: float, widening_alpha: float) -> bool: # Checks wether state can or cannot expand, based on availability of untried moves and child limit
        """Progressive widening keeps a wide feature pool searchable."""
        if not self.untried_moves:
            return False
        child_limit = max(
            1,
            int(math.ceil(widening_constant * max(1, self.visits) ** widening_alpha)),
        )
        return len(self.children) < child_limit

    def ucb1_score(self, exploration_constant: float = math.sqrt(2.0)) -> float: # Calculates selection priority of child
        if self.visits == 0: # if child is unvisited it gets infinite priority, so that it doesn't get ignored and should be chosen right away
            return float("inf")
        if self.parent is None: # if node is parent then we just take its mean reward because there is no parent visits to calculate with full formula
            return self.mean_reward
        exploration = exploration_constant * math.sqrt(
            math.log(max(1, self.parent.visits)) / self.visits
        ) 
        return self.mean_reward + exploration # If child is visited less then its priority grows, on the other hand if the child is visited often then its priority decreses

    def best_child(self, exploration_constant: float) -> "MCTSNode": # Returns best child based on the highest UCB1 score
        return max(self.children, key=lambda c: c.ucb1_score(exploration_constant))

    def expand(self, rng: random.Random) -> "MCTSNode": # Creates new child, appends it to current node's child list and removes the chosen move from untried moves
        move = self.untried_moves.pop()
        child = MCTSNode(self.game_state.make_move(move), self, move, rng)
        self.children.append(child)
        return child

    def update(self, reward: float) -> None: # Updates node's values after backpropogation, incrementing visits and adding up to total reward
        self.visits += 1
        self.total_reward += reward


@dataclass(frozen=True)
class MCTSResult:
    best_state: FeatureSelectionState
    best_reward: float
    root: MCTSNode
    reward_history: tuple[float, ...]
    best_reward_history: tuple[float, ...]


IterationCallback = Callable[ # Function called after every every interation, used by main.py to use the values for printing stats of run and taking snapshots
    [int, int, MCTSNode, FeatureSelectionState, float, FeatureSelectionState, float],
    None,
]


def _rollout(state: FeatureSelectionState, rng: random.Random) -> FeatureSelectionState: # Performs simulation, calculates remaining moves and needed feature, then using random sampling fills state's children
    """Randomly complete a partial subset without building temporary tree nodes."""
    if state.is_terminal():
        return state
    remaining = state.get_legal_moves()
    needed = state.subset_size - len(state.selected)
    selected = state.selected + tuple(rng.sample(remaining, needed))
    return FeatureSelectionState(
        state.n_features, state.subset_size, selected, state.feature_names
    )


def _backpropagate(node: MCTSNode, reward: float) -> None:
    while node is not None: # Check that it reached parent or not, moves backward until it reaches root
        node.update(reward) # Updates the reward of the node
        node = node.parent # Moves up, to the parent 


def mcts_search( # Main search function connection all internal functions
    initial_state: FeatureSelectionState,
    evaluate: Callable[[FeatureSelectionState], float],
    n_iterations: int = 500,
    exploration_constant: float = math.sqrt(2.0),
    widening_constant: float = 1.5,
    widening_alpha: float = 0.5,
    random_state: int = 42,
    on_iteration: Optional[IterationCallback] = None,
) -> MCTSResult:
    """Search for the terminal subset with the greatest evaluator reward.

    ``evaluate`` is called only for complete subsets and should return a bounded
    reward where larger is better. Progressive widening is important here:
    without it, hundreds of candidate features keep the root from ever growing
    deeper during a practical number of iterations.
    """
    if n_iterations < 1:
        raise ValueError("n_iterations must be at least 1")
    if initial_state.is_terminal(): # Handles already terminal state
        reward = float(evaluate(initial_state))
        root = MCTSNode(initial_state, rng=random.Random(random_state))
        root.update(reward)
        return MCTSResult(initial_state, reward, root, (reward,), (reward,))

    rng = random.Random(random_state)
    root = MCTSNode(initial_state, rng=rng)
    best_state: Optional[FeatureSelectionState] = None
    best_reward = float("-inf")
    history: list[float] = []
    best_history: list[float] = []

    for iteration in range(1, n_iterations + 1):
        node = root

        while not node.is_terminal():
            if node.can_expand(widening_constant, widening_alpha):
                node = node.expand(rng)
                break
            if not node.children:  # defensive; a valid non-terminal state has moves
                break
            node = node.best_child(exploration_constant)

        terminal_state = _rollout(node.game_state, rng)
        reward = float(evaluate(terminal_state))
        if not math.isfinite(reward):
            raise ValueError(f"evaluate returned a non-finite reward: {reward}")
        _backpropagate(node, reward)

        if reward > best_reward:
            best_reward = reward
            best_state = terminal_state
        history.append(reward)
        best_history.append(best_reward)

        if on_iteration is not None:
            on_iteration(
                iteration,
                n_iterations,
                node,
                terminal_state,
                reward,
                best_state,
                best_reward,
            )

    return MCTSResult(
        best_state=best_state,
        best_reward=best_reward,
        root=root,
        reward_history=tuple(history),
        best_reward_history=tuple(best_history),
    )
