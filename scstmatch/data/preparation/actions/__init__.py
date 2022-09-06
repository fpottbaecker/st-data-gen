__all__ = ["Action", "actions", "FilterAction", "FindMarkers", "SplitAction"]

from .action import Action

from .filter import FilterAction
from .markers import FindMarkers
from .split import SplitAction

actions = {
    "filter": FilterAction,
    "split": SplitAction,
    "markers": FindMarkers,
}
