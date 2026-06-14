"""Verify all modules import correctly."""

import yaml
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_config_loads():
    with open('config/config.yaml') as f:
        config = yaml.safe_load(f)
    assert 'data' in config
    assert 'features' in config
    assert 'models' in config
    assert 'ensemble' in config
    print("✓ config.yaml loads successfully")


def test_src_imports():
    from src.data.loader import MatchDataLoader
    print("✓ src.data.loader")
    from src.features.builder import FeatureBuilder
    print("✓ src.features.builder")
    from src.models.ml_models import MLModelFactory, NeuralNet, TorchClassifier
    print("✓ src.models.ml_models")
    from src.models.statistical import PoissonModel, EloModel, MonteCarloSimulator
    print("✓ src.models.statistical")
    from src.models.ensemble import HybridEnsemble
    print("✓ src.models.ensemble")
    from src.evaluation.metrics import Evaluator
    print("✓ src.evaluation.metrics")
    from src.simulation.simulator import GroupStage, TournamentSimulator, world_cup_2026_groups
    print("✓ src.simulation.simulator")
    from src.pipeline import FIFAPredictor
    print("✓ src.pipeline")
    print("\n🎯 All imports successful!")


if __name__ == '__main__':
    test_config_loads()
    test_src_imports()
