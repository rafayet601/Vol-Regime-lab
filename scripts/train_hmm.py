#!/usr/bin/env python3
"""
Training script for S&P 500 Student-t HMM with Baum-Welch algorithm.

v2.0 — Uses the functional feature API (build_features, get_feature_cols)
instead of the legacy FeatureEngineer class.
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_lab.data.features import (
    build_features,
    get_feature_cols,
    load_spx_data,
    load_vix_data,
)
from regime_lab.models.hmm_studentt import StudentTHMM, select_n_states
from regime_lab.utils.config import ensure_dir, get_timestamp, save_json
from regime_lab.utils.io import save_dataframe, save_pickle


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("regime_lab.log")
        ]
    )


def load_and_prepare_data(
    symbol: str = "^GSPC",
    start_date: str = "2005-01-01",
    end_date: Optional[str] = None,
    tier: int = 2,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load and prepare S&P 500 data with v2.0 feature engineering.

    Parameters
    ----------
    symbol : str
        Ticker symbol (only ^GSPC supported by `load_spx_data`).
    start_date, end_date : str
        Date range for data retrieval.
    tier : int
        Feature tier (1=SPX only, 2=+VIX, 3=full).

    Returns
    -------
    features : DataFrame
        Feature matrix with all tiers available.
    feature_cols : list of str
        Selected feature column names for this tier.
    """
    logger = logging.getLogger(__name__)

    # --- Load SPX prices + returns ---
    logger.info("Loading S&P 500 data...")
    prices, returns = load_spx_data(start_date=start_date, end_date=end_date)

    # --- Optionally load VIX data for Tier 2+ features ---
    vix_df = None
    if tier >= 2:
        logger.info("Loading VIX data for Tier 2+ features...")
        try:
            vix_df = load_vix_data(start_date=start_date, end_date=end_date)
        except Exception as exc:
            logger.warning(f"VIX data unavailable ({exc}); falling back to Tier 1.")
            tier = 1

    # --- Build feature matrix ---
    logger.info("Building feature matrix...")
    features = build_features(returns, prices=prices, vix_df=vix_df)
    feature_cols = get_feature_cols(features, tier=tier)

    logger.info(
        f"Data preparation complete: {len(features)} observations, "
        f"{len(feature_cols)} features (Tier {tier})"
    )
    return features, feature_cols


def train_hmm_model(
    data: pd.DataFrame,
    feature_columns: List[str],
    n_states: int = 2,
    fix_nu: bool = False,
    max_iter: int = 200,
    tol: float = 1e-6,
    random_seed: int = 42,
) -> StudentTHMM:
    """Train the Student-t HMM model.

    Parameters
    ----------
    data : DataFrame
        Feature matrix produced by build_features().
    feature_columns : list of str
        Which columns to feed into the HMM.
    n_states, fix_nu, max_iter, tol, random_seed
        StudentTHMM constructor parameters.

    Returns
    -------
    Trained StudentTHMM instance.
    """
    logger = logging.getLogger(__name__)

    X = data[feature_columns].values

    logger.info(f"Initializing Student-t HMM (K={n_states}, fix_nu={fix_nu})...")
    model = StudentTHMM(
        n_states=n_states,
        fix_nu=fix_nu,
        max_iter=max_iter,
        tol=tol,
        random_seed=random_seed,
    )

    logger.info("Training HMM with Baum-Welch algorithm...")
    start = time.time()
    model.fit(X)
    elapsed = time.time() - start

    logger.info(
        f"HMM training completed in {elapsed:.1f}s "
        f"(converged={model.params_.converged}, iter={model.params_.n_iter})"
    )
    return model


def generate_predictions(
    model: StudentTHMM,
    data: pd.DataFrame,
    feature_columns: List[str],
) -> Dict[str, pd.DataFrame]:
    """Generate predictions and probabilities from the trained model.

    Returns
    -------
    dict with keys 'predictions' and 'features'.
    """
    logger = logging.getLogger(__name__)

    X = data[feature_columns].values

    logger.info("Generating Viterbi state predictions...")
    predicted_states = model.predict(X)

    logger.info("Computing posterior state probabilities...")
    state_probabilities = model.predict_proba(X)

    K = state_probabilities.shape[1]
    predictions_df = pd.DataFrame(
        {"date": data.index, "predicted_state": predicted_states},
    )
    for k in range(K):
        predictions_df[f"state_{k}_prob"] = state_probabilities[:, k]

    features_df = data[feature_columns].copy()
    features_df.index.name = "date"

    results = {
        "predictions": predictions_df,
        "features": features_df,
    }

    logger.info(f"Generated predictions for {len(predictions_df)} observations")
    return results


def save_results(
    model: StudentTHMM,
    predictions: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    output_dir: str,
) -> None:
    """Save training results and model artifacts."""
    logger = logging.getLogger(__name__)

    ensure_dir(output_dir)

    # Model parameters JSON
    model_summary = model.get_model_summary()
    model_summary["feature_cols"] = feature_cols
    model_summary["timestamp"] = get_timestamp()

    params_file = Path(output_dir) / "params.json"
    save_json(model_summary, str(params_file))
    logger.info(f"Saved model parameters to {params_file}")

    # Predictions CSV (posteriors.csv — expected by plot_regimes / evaluate)
    predictions_file = Path(output_dir) / "posteriors.csv"
    save_dataframe(predictions["predictions"], str(predictions_file))
    logger.info(f"Saved predictions to {predictions_file}")

    # Features CSV
    features_file = Path(output_dir) / "features.csv"
    save_dataframe(predictions["features"], str(features_file))
    logger.info(f"Saved features to {features_file}")

    # Viterbi path CSV
    viterbi_file = Path(output_dir) / "viterbi.csv"
    viterbi_df = predictions["predictions"][
        ["date", "predicted_state"]
        + [c for c in predictions["predictions"].columns if c.endswith("_prob")]
    ].copy()
    save_dataframe(viterbi_df, str(viterbi_file))
    logger.info(f"Saved Viterbi path to {viterbi_file}")

    # Run summary JSON
    run_summary = {
        "timestamp": get_timestamp(),
        "model_type": "Student-t HMM",
        "n_states": model.n_states,
        "n_observations": len(predictions["predictions"]),
        "n_features": len(predictions["features"].columns),
        "feature_cols": feature_cols,
        "log_likelihood": model.params_.log_likelihood,
        "bic": model.params_.bic,
        "converged": model.params_.converged,
        "n_iter": model.params_.n_iter,
        "files": {
            "params": str(params_file),
            "posteriors": str(predictions_file),
            "features": str(features_file),
            "viterbi": str(viterbi_file),
        },
    }

    summary_file = Path(output_dir) / "last_run.json"
    save_json(run_summary, str(summary_file))
    logger.info(f"Saved run summary to {summary_file}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train S&P 500 Student-t HMM")
    parser.add_argument(
        "--start-date", type=str, default="2005-01-01",
        help="Start date for data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="End date for data (YYYY-MM-DD, default: today)"
    )
    parser.add_argument(
        "--tier", type=int, default=2, choices=[1, 2, 3],
        help="Feature tier (1=SPX, 2=+VIX, 3=full)"
    )
    parser.add_argument(
        "--n-states", type=int, default=2,
        help="Number of hidden states K"
    )
    parser.add_argument(
        "--auto-select-k", action="store_true",
        help="Automatically select K via BIC from {2,3,4}"
    )
    parser.add_argument(
        "--fix-nu", action="store_true",
        help="Fix degrees of freedom (do not learn ν)"
    )
    parser.add_argument(
        "--max-iter", type=int, default=200,
        help="Maximum EM iterations"
    )
    parser.add_argument(
        "--tol", type=float, default=1e-6,
        help="Convergence tolerance"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output-dir", type=str, default="artifacts",
        help="Output directory for artifacts"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting HMM training pipeline (v2.0)...")

        # Load and prepare data
        features, feature_cols = load_and_prepare_data(
            start_date=args.start_date,
            end_date=args.end_date,
            tier=args.tier,
        )

        # Determine number of states
        n_states = args.n_states
        if args.auto_select_k:
            logger.info("Auto-selecting K via BIC...")
            X_sel = features[feature_cols].values
            n_states, bic_val = select_n_states(
                X_sel, k_range=[2, 3, 4], criterion="bic",
                fix_nu=args.fix_nu, max_iter=min(args.max_iter, 100),
                random_seed=args.seed,
            )
            logger.info(f"Selected K={n_states} (BIC={bic_val:.2f})")

        # Train model
        model = train_hmm_model(
            features, feature_cols,
            n_states=n_states,
            fix_nu=args.fix_nu,
            max_iter=args.max_iter,
            tol=args.tol,
            random_seed=args.seed,
        )

        # Generate predictions
        predictions = generate_predictions(model, features, feature_cols)

        # Save results
        save_results(model, predictions, feature_cols, args.output_dir)

        logger.info("HMM training pipeline completed successfully!")

        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        p = model.params_
        print(f"Model:          Student-t HMM")
        print(f"States:         {model.n_states}")
        print(f"Features:       {len(feature_cols)} (Tier {args.tier})")
        print(f"  Columns:      {feature_cols}")
        print(f"Observations:   {len(features)}")
        print(f"Converged:      {p.converged} (iter {p.n_iter})")
        print(f"Log-likelihood: {p.log_likelihood:.2f}")
        print(f"BIC:            {p.bic:.2f}")
        print(f"ν (d.f.):       {p.nu}")
        print(f"Output:         {args.output_dir}/")
        print("="*60)

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
