from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class FeatureSelectionState:
    """An immutable, hashable node in the feature-selection search space."""

    n_features: int
    subset_size: int
    selected: Tuple[int, ...] = ()
    feature_names: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if not 0 < self.subset_size <= self.n_features:
            raise ValueError("subset_size must be between 1 and n_features")
        if len(self.selected) > self.subset_size:
            raise ValueError("selected contains more features than subset_size")
        if len(set(self.selected)) != len(self.selected):
            raise ValueError("selected feature indices must be unique")
        if any(i < 0 or i >= self.n_features for i in self.selected):
            raise ValueError("selected contains an out-of-range feature index")
        if self.feature_names is not None and len(self.feature_names) != self.n_features:
            raise ValueError("feature_names length must equal n_features")

    def get_legal_moves(self) -> list[int]: # Checks which features can be added next
        if self.is_terminal():
            return []
        chosen = set(self.selected)
        return [i for i in range(self.n_features) if i not in chosen] # Gets all possible moves within n_features range and then removes already chosen ones

    def make_move(self, feature_index: int) -> "FeatureSelectionState": # Adds new feature to selected list 
        if feature_index not in self.get_legal_moves(): # Checks wether feature_index is in legal moves list
            raise ValueError(f"Illegal feature index {feature_index}")
        return FeatureSelectionState( # Constructs new successor state with new list of selected, everything else is inherited
            n_features=self.n_features,
            subset_size=self.subset_size,
            selected=self.selected + (feature_index,),
            feature_names=self.feature_names,
        )

    def is_terminal(self) -> bool: # Checks wether state is terminal, meaning number of selected and subset_size equals, so there is no more space for expansion
        return len(self.selected) == self.subset_size

    def get_state(self) -> Tuple[int, ...]: # During training order of the states doesn't matter, but there can appear duplicates just with other order of the same states
        return tuple(sorted(self.selected)) # that's why we order the selected to clearly see that it is not the same list of states, and if it is use the values of already evaluated version

    def feature_label(self, feature_index: int) -> str: # Converts feature index into readable name for terminal printing
        if self.feature_names is None:
            return f"feature_{feature_index}"
        return self.feature_names[feature_index]

    def selected_labels(self) -> list[str]: # Returns list of names of selected features
        return [self.feature_label(i) for i in self.selected]


def make_initial_state( # Constructs root state, meaning first empty state
    feature_names: Sequence[str], subset_size: int
) -> FeatureSelectionState:
    names = tuple(str(name) for name in feature_names)
    return FeatureSelectionState(len(names), subset_size, feature_names=names)