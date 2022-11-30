__all__ = ["Action", "actions", "FilterAction", "FindMarkers", "SplitAction"]

from .action import Action

from .filter import FilterAction
from .markers import FindMarkers
from .output_reference import OutputReference
from .split import SplitAction
from .synthesize_spatial import SynthesizeSpatial

actions = {
    "filter": FilterAction,
    "split": SplitAction,
    "markers": FindMarkers,
    "output_reference": OutputReference,
    "synthesize_spatial": SynthesizeSpatial,
}
