"""
Vol Regime Lab — backtest sub-package.
"""
from .walk_forward import WalkForwardBacktester, WalkForwardResult, print_backtest_summary
from .garch_benchmark import GARCHBenchmark, GARCHBenchmarkResult, run_garch_benchmark, print_garch_comparison
from .strategy import (
    ThresholdStrategy,
    ProbabilityWeightedStrategy,
    VolTargetingStrategy,
    RegimeSwitchingStrategy,
    build_strategy_suite,
)

__all__ = [
    "WalkForwardBacktester",
    "WalkForwardResult",
    "print_backtest_summary",
    "GARCHBenchmark",
    "GARCHBenchmarkResult",
    "run_garch_benchmark",
    "print_garch_comparison",
    "ThresholdStrategy",
    "ProbabilityWeightedStrategy",
    "VolTargetingStrategy",
    "RegimeSwitchingStrategy",
    "build_strategy_suite",
]
