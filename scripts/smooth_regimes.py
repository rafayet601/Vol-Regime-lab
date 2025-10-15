#!/usr/bin/env python3
"""Post-process regime predictions to enforce minimum regime duration."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_lab.utils.config import ensure_dir, save_json
from regime_lab.utils.io import load_dataframe, save_dataframe


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def smooth_regime_sequence(
    states: np.ndarray,
    min_duration: int = 5,
    smoothing_method: str = "fill"
) -> np.ndarray:
    """Smooth regime sequence to enforce minimum duration.
    
    Args:
        states: Array of regime states (0 or 1)
        min_duration: Minimum duration for a regime (in days)
        smoothing_method: Method to use ('fill', 'modal', or 'median')
        
    Returns:
        Smoothed regime sequence
    """
    if smoothing_method == "fill":
        return _fill_short_regimes(states, min_duration)
    elif smoothing_method == "modal":
        return _modal_smoothing(states, min_duration)
    elif smoothing_method == "median":
        return _median_filter(states, min_duration)
    else:
        raise ValueError(f"Unknown smoothing method: {smoothing_method}")


def _fill_short_regimes(states: np.ndarray, min_duration: int) -> np.ndarray:
    """Fill short regimes by merging them with the previous regime.
    
    This method identifies regime periods shorter than min_duration and
    replaces them with the previous regime's state.
    
    Args:
        states: Original regime sequence
        min_duration: Minimum allowed regime duration
        
    Returns:
        Smoothed regime sequence
    """
    smoothed = states.copy()
    
    # Keep iterating until no more short regimes exist
    changed = True
    max_iterations = 100
    iteration = 0
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        i = 0
        
        while i < len(smoothed):
            # Find the length of current regime
            current_state = smoothed[i]
            j = i
            while j < len(smoothed) and smoothed[j] == current_state:
                j += 1
            
            regime_length = j - i
            
            # If regime is too short, fill it with the previous regime's state
            if regime_length < min_duration and i > 0:
                prev_state = smoothed[i - 1]
                smoothed[i:j] = prev_state
                changed = True
                # Start over from beginning after making a change
                break
            
            i = j
    
    return smoothed


def _modal_smoothing(states: np.ndarray, window: int) -> np.ndarray:
    """Apply modal (most common) filtering with a rolling window.
    
    Args:
        states: Original regime sequence
        window: Window size for modal filter
        
    Returns:
        Smoothed regime sequence
    """
    smoothed = np.zeros_like(states)
    half_window = window // 2
    
    for i in range(len(states)):
        start = max(0, i - half_window)
        end = min(len(states), i + half_window + 1)
        window_states = states[start:end]
        
        # Use mode (most common state in window)
        smoothed[i] = np.bincount(window_states).argmax()
    
    return smoothed


def _median_filter(states: np.ndarray, window: int) -> np.ndarray:
    """Apply median filter with a rolling window.
    
    Args:
        states: Original regime sequence
        window: Window size for median filter
        
    Returns:
        Smoothed regime sequence
    """
    smoothed = np.zeros_like(states)
    half_window = window // 2
    
    for i in range(len(states)):
        start = max(0, i - half_window)
        end = min(len(states), i + half_window + 1)
        window_states = states[start:end]
        
        # Use median
        smoothed[i] = int(np.median(window_states))
    
    return smoothed


def compute_regime_statistics(states: np.ndarray) -> dict:
    """Compute statistics about regime sequence.
    
    Args:
        states: Regime sequence
        
    Returns:
        Dictionary of statistics
    """
    from itertools import groupby
    
    transitions = np.sum(np.diff(states) != 0)
    runs = [(state, len(list(group))) for state, group in groupby(states)]
    run_lengths = [length for _, length in runs]
    
    stats = {
        "total_days": len(states),
        "transitions": int(transitions),
        "num_periods": len(runs),
        "avg_duration": float(np.mean(run_lengths)),
        "median_duration": float(np.median(run_lengths)),
        "min_duration": int(np.min(run_lengths)),
        "max_duration": int(np.max(run_lengths)),
        "state_0_days": int(np.sum(states == 0)),
        "state_1_days": int(np.sum(states == 1)),
        "state_0_pct": float(np.mean(states == 0) * 100),
        "state_1_pct": float(np.mean(states == 1) * 100),
    }
    
    return stats


def main():
    """Main smoothing function."""
    parser = argparse.ArgumentParser(description="Smooth regime predictions")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="artifacts",
        help="Directory containing model artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts_smoothed",
        help="Output directory for smoothed results"
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=10,
        help="Minimum regime duration in days"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["fill", "modal", "median"],
        default="fill",
        help="Smoothing method"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting regime smoothing pipeline...")
        
        # Load predictions
        input_path = Path(args.input_dir)
        predictions_file = input_path / "posteriors.csv"
        
        if not predictions_file.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
        
        predictions = load_dataframe(str(predictions_file))
        logger.info(f"Loaded predictions from {predictions_file}")
        
        # Get original statistics
        original_states = predictions['predicted_state'].values
        orig_stats = compute_regime_statistics(original_states)
        
        logger.info(f"Original: {orig_stats['transitions']} transitions, "
                   f"avg duration {orig_stats['avg_duration']:.2f} days")
        
        # Smooth the regime sequence
        logger.info(f"Applying {args.method} smoothing with min_duration={args.min_duration}...")
        smoothed_states = smooth_regime_sequence(
            original_states,
            min_duration=args.min_duration,
            smoothing_method=args.method
        )
        
        # Get smoothed statistics
        smooth_stats = compute_regime_statistics(smoothed_states)
        logger.info(f"Smoothed: {smooth_stats['transitions']} transitions, "
                   f"avg duration {smooth_stats['avg_duration']:.2f} days")
        
        # Update predictions with smoothed states
        smoothed_predictions = predictions.copy()
        smoothed_predictions['predicted_state'] = smoothed_states
        
        # Note: probabilities are kept from original model (they represent uncertainty)
        # The smoothed_state is a post-processed decision
        
        # Save smoothed results
        ensure_dir(args.output_dir)
        output_file = Path(args.output_dir) / "posteriors.csv"
        save_dataframe(smoothed_predictions, str(output_file))
        logger.info(f"Saved smoothed predictions to {output_file}")
        
        # Save statistics comparison
        comparison = {
            "original": orig_stats,
            "smoothed": smooth_stats,
            "smoothing_params": {
                "method": args.method,
                "min_duration": args.min_duration
            }
        }
        stats_file = Path(args.output_dir) / "smoothing_stats.json"
        save_json(comparison, str(stats_file))
        logger.info(f"Saved statistics to {stats_file}")
        
        # Copy other files
        for filename in ["features.csv", "viterbi.csv", "params.json", "model.pkl", "last_run.json"]:
            src = input_path / filename
            if src.exists():
                import shutil
                dst = Path(args.output_dir) / filename
                shutil.copy2(src, dst)
                logger.info(f"Copied {filename} to output directory")
        
        # Print summary
        print("\n" + "="*70)
        print("REGIME SMOOTHING SUMMARY")
        print("="*70)
        print(f"Method: {args.method}")
        print(f"Minimum duration: {args.min_duration} days")
        print()
        print("BEFORE SMOOTHING:")
        print(f"  Transitions: {orig_stats['transitions']}")
        print(f"  Average duration: {orig_stats['avg_duration']:.2f} days")
        print(f"  Median duration: {orig_stats['median_duration']} days")
        print(f"  Max duration: {orig_stats['max_duration']} days")
        print()
        print("AFTER SMOOTHING:")
        print(f"  Transitions: {smooth_stats['transitions']}")
        print(f"  Average duration: {smooth_stats['avg_duration']:.2f} days")
        print(f"  Median duration: {smooth_stats['median_duration']} days")
        print(f"  Max duration: {smooth_stats['max_duration']} days")
        print()
        print(f"Transitions reduced by: {orig_stats['transitions'] - smooth_stats['transitions']} "
              f"({(1 - smooth_stats['transitions']/orig_stats['transitions'])*100:.1f}%)")
        print(f"Duration increased by: {smooth_stats['avg_duration'] / orig_stats['avg_duration']:.2f}x")
        print("="*70)
        
        logger.info("Regime smoothing pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Smoothing pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()

