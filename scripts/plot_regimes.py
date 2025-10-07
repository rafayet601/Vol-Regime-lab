#!/usr/bin/env python3
"""Plotting script for regime detection results."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_lab.plotting.regimes import RegimePlotter, plot_regimes_from_artifacts
from regime_lab.utils.config import ensure_dir, load_json
from regime_lab.utils.io import load_dataframe


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )


def load_artifacts(artifacts_dir: str):
    """Load model artifacts from directory.
    
    Args:
        artifacts_dir: Directory containing artifacts
        
    Returns:
        Tuple of (run_summary, predictions, features)
    """
    logger = logging.getLogger(__name__)
    
    artifacts_path = Path(artifacts_dir)
    
    # Load run summary
    summary_file = artifacts_path / "last_run.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Run summary not found: {summary_file}")
    
    run_summary = load_json(str(summary_file))
    logger.info(f"Loaded run summary from {summary_file}")
    
    # Load predictions
    predictions_file = artifacts_path / "posteriors.csv"
    if not predictions_file.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_file}")
    
    predictions = load_dataframe(str(predictions_file))
    logger.info(f"Loaded predictions: {len(predictions)} observations")
    
    # Load features
    features_file = artifacts_path / "features.csv"
    if not features_file.exists():
        logger.warning(f"Features file not found: {features_file}")
        features = None
    else:
        features = load_dataframe(str(features_file))
        logger.info(f"Loaded features: {len(features)} observations, {len(features.columns)} features")
    
    return run_summary, predictions, features


def create_plots(
    run_summary: dict,
    predictions,
    features,
    output_dir: str,
    plot_type: Optional[str] = None
) -> None:
    """Create regime plots based on available data.
    
    Args:
        run_summary: Run summary dictionary
        predictions: Predictions DataFrame
        features: Features DataFrame (optional)
        output_dir: Output directory for plots
        plot_type: Type of plot to create (optional)
    """
    logger = logging.getLogger(__name__)
    
    ensure_dir(output_dir)
    plotter = RegimePlotter()
    
    logger.info(f"Creating plots in {output_dir}")
    
    if plot_type is None or plot_type == "all":
        # Create all available plots
        
        # 1. Regime transitions
        plotter.plot_regime_transitions(
            predictions,
            title=f"Regime Transitions - {run_summary.get('model_type', 'HMM')}",
            save_path=f"{output_dir}/regime_transitions.png"
        )
        
        # 2. Feature analysis (if features available)
        if features is not None:
            feature_columns = [col for col in features.columns 
                              if col not in ['returns', 'date', 'datetime']]
            
            if feature_columns:
                plotter.plot_feature_analysis(
                    features, predictions, feature_columns,
                    title="Feature Analysis by Regime",
                    save_path=f"{output_dir}/feature_analysis.png"
                )
            
            # 3. Volatility comparison (if volatility features available)
            volatility_cols = [col for col in features.columns 
                              if 'vol' in col.lower() or 'std' in col.lower()]
            if volatility_cols:
                combined_data = features.copy()
                combined_data['predicted_state'] = predictions['predicted_state'].values
                
                plotter.plot_volatility_comparison(
                    combined_data,
                    title="Volatility Measures Comparison",
                    save_path=f"{output_dir}/volatility_comparison.png"
                )
        
        logger.info("Created comprehensive regime analysis plots")
    
    elif plot_type == "transitions":
        plotter.plot_regime_transitions(
            predictions,
            title=f"Regime Transitions - {run_summary.get('model_type', 'HMM')}",
            save_path=f"{output_dir}/regime_transitions.png"
        )
        logger.info("Created regime transitions plot")
    
    elif plot_type == "features" and features is not None:
        feature_columns = [col for col in features.columns 
                          if col not in ['returns', 'date', 'datetime']]
        
        if feature_columns:
            plotter.plot_feature_analysis(
                features, predictions, feature_columns,
                title="Feature Analysis by Regime",
                save_path=f"{output_dir}/feature_analysis.png"
            )
            logger.info("Created feature analysis plot")
        else:
            logger.warning("No feature columns found for analysis")
    
    elif plot_type == "volatility" and features is not None:
        volatility_cols = [col for col in features.columns 
                          if 'vol' in col.lower() or 'std' in col.lower()]
        
        if volatility_cols:
            combined_data = features.copy()
            combined_data['predicted_state'] = predictions['predicted_state'].values
            
            plotter.plot_volatility_comparison(
                combined_data,
                title="Volatility Measures Comparison",
                save_path=f"{output_dir}/volatility_comparison.png"
            )
            logger.info("Created volatility comparison plot")
        else:
            logger.warning("No volatility columns found for comparison")
    
    else:
        logger.error(f"Unknown plot type: {plot_type}")


def print_summary(run_summary: dict, predictions) -> None:
    """Print run summary information.
    
    Args:
        run_summary: Run summary dictionary
        predictions: Predictions DataFrame
    """
    print("\n" + "="*60)
    print("REGIME DETECTION RESULTS SUMMARY")
    print("="*60)
    print(f"Model Type: {run_summary.get('model_type', 'Unknown')}")
    print(f"Number of States: {run_summary.get('n_states', 'Unknown')}")
    print(f"Observations: {len(predictions)}")
    print(f"Features: {run_summary.get('n_features', 'Unknown')}")
    print(f"Timestamp: {run_summary.get('timestamp', 'Unknown')}")
    
    # Regime statistics
    if 'predicted_state' in predictions.columns:
        regime_counts = predictions['predicted_state'].value_counts().sort_index()
        regime_pct = predictions['predicted_state'].value_counts(normalize=True).sort_index()
        
        print("\nRegime Distribution:")
        for state, count in regime_counts.items():
            pct = regime_pct[state] * 100
            regime_name = "Low Volatility" if state == 0 else "High Volatility"
            print(f"  State {state} ({regime_name}): {count} observations ({pct:.1f}%)")
    
    # Transition statistics
    if len(predictions) > 1:
        transitions = (predictions['predicted_state'].diff() != 0).sum()
        print(f"\nTotal Regime Transitions: {transitions}")
        print(f"Average Regime Duration: {len(predictions) / (transitions + 1):.1f} observations")
    
    print("="*60)


def main():
    """Main plotting function."""
    parser = argparse.ArgumentParser(description="Plot regime detection results")
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory containing model artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/figures",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        choices=["all", "transitions", "features", "volatility"],
        default="all",
        help="Type of plot to create"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print summary only, don't create plots"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting regime plotting pipeline...")
        
        # Load artifacts
        run_summary, predictions, features = load_artifacts(args.artifacts_dir)
        
        # Print summary
        print_summary(run_summary, predictions)
        
        if not args.summary_only:
            # Create plots
            create_plots(
                run_summary, predictions, features, 
                args.output_dir, args.plot_type
            )
            logger.info(f"Plots saved to {args.output_dir}")
        
        logger.info("Plotting pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Plotting pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
