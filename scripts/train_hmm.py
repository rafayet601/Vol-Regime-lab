#!/usr/bin/env python3
"""Training script for S&P 500 Student-t HMM with Baum-Welch algorithm."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from regime_lab.data.features import FeatureEngineer
from regime_lab.data.loader import SPXDataLoader
from regime_lab.models.hmm_studentt import StudentTHMM
from regime_lab.utils.config import ensure_dir, get_timestamp, load_config, save_json
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


def load_and_prepare_data(config: Dict) -> pd.DataFrame:
    """Load and prepare S&P 500 data with features.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        DataFrame with engineered features
    """
    logger = logging.getLogger(__name__)
    
    # Initialize data loader
    loader = SPXDataLoader(cache_dir=config["data"]["cache_dir"])
    
    # Load price and returns data
    logger.info("Loading S&P 500 data...")
    price_data, returns_data = loader.get_full_dataset(
        symbol=config["data"]["symbol"],
        start_date=config["data"]["start_date"],
        end_date=config["data"]["end_date"]
    )
    
    # Engineer features
    logger.info("Engineering features...")
    feature_engineer = FeatureEngineer(
        returns_column=config["features"]["returns_column"]
    )
    
    feature_data = feature_engineer.engineer_features(
        returns_data,
        rolling_window=config["features"]["rolling_window"],
        additional_features=config["features"]["additional_features"]
    )
    
    # Validate features
    feature_columns = feature_engineer.get_feature_names(feature_data)
    if not feature_engineer.validate_features(feature_data, feature_columns):
        raise ValueError("Feature validation failed")
    
    logger.info(f"Data preparation complete: {len(feature_data)} observations, {len(feature_columns)} features")
    
    return feature_data, feature_columns


def train_hmm_model(
    data: pd.DataFrame,
    feature_columns: list,
    config: Dict
) -> StudentTHMM:
    """Train the Student-t HMM model.
    
    Args:
        data: DataFrame with features
        feature_columns: List of feature column names
        config: Configuration dictionary
        
    Returns:
        Trained HMM model
    """
    logger = logging.getLogger(__name__)
    
    # Prepare training data
    X = data[feature_columns].values
    
    # Initialize and train model
    logger.info("Initializing Student-t HMM...")
    model = StudentTHMM(
        n_states=config["model"]["n_states"],
        n_features=len(feature_columns),
        df=5.0,  # Fixed degrees of freedom
        random_seed=config["training"]["random_seed"]
    )
    
    logger.info("Training HMM with Baum-Welch algorithm...")
    model.fit(
        X,
        max_iterations=config["training"]["max_iterations"],
        tolerance=config["training"]["tolerance"],
        verbose=True
    )
    
    logger.info("HMM training completed successfully")
    return model


def generate_predictions(
    model: StudentTHMM,
    data: pd.DataFrame,
    feature_columns: list
) -> Dict[str, pd.DataFrame]:
    """Generate predictions and probabilities from the trained model.
    
    Args:
        model: Trained HMM model
        data: DataFrame with features
        feature_columns: List of feature column names
        
    Returns:
        Dictionary containing prediction results
    """
    logger = logging.getLogger(__name__)
    
    # Prepare data
    X = data[feature_columns].values
    
    # Generate predictions
    logger.info("Generating state predictions...")
    predicted_states = model.predict_states(X)
    
    logger.info("Computing state probabilities...")
    state_probabilities = model.predict_proba(X)
    
    # Create results DataFrames
    predictions_df = pd.DataFrame({
        "date": data.index,
        "predicted_state": predicted_states,
        "state_0_prob": state_probabilities[:, 0],
        "state_1_prob": state_probabilities[:, 1]
    })
    
    # Add feature data for analysis
    features_df = data[feature_columns].copy()
    features_df.index.name = "date"
    
    results = {
        "predictions": predictions_df,
        "features": features_df
    }
    
    logger.info(f"Generated predictions for {len(predictions_df)} observations")
    
    return results


def save_results(
    model: StudentTHMM,
    predictions: Dict[str, pd.DataFrame],
    config: Dict,
    output_dir: str
) -> None:
    """Save training results and model artifacts.
    
    Args:
        model: Trained HMM model
        predictions: Prediction results
        config: Configuration dictionary
        output_dir: Output directory for artifacts
    """
    logger = logging.getLogger(__name__)
    
    # Ensure output directory exists
    ensure_dir(output_dir)
    
    # Save model parameters
    model_summary = model.get_model_summary()
    model_summary["training_config"] = config
    model_summary["timestamp"] = get_timestamp()
    
    params_file = Path(output_dir) / "params.json"
    save_json(model_summary, str(params_file))
    logger.info(f"Saved model parameters to {params_file}")
    
    # Save model object
    model_file = Path(output_dir) / "model.pkl"
    model.save_model(str(model_file))
    logger.info(f"Saved model object to {model_file}")
    
    # Save predictions
    predictions_file = Path(output_dir) / "posteriors.csv"
    save_dataframe(predictions["predictions"], str(predictions_file))
    logger.info(f"Saved predictions to {predictions_file}")
    
    # Save features
    features_file = Path(output_dir) / "features.csv"
    save_dataframe(predictions["features"], str(features_file))
    logger.info(f"Saved features to {features_file}")
    
    # Save Viterbi path (state sequence)
    viterbi_file = Path(output_dir) / "viterbi.csv"
    viterbi_df = pd.DataFrame({
        "date": predictions["predictions"]["date"],
        "state": predictions["predictions"]["predicted_state"],
        "state_0_prob": predictions["predictions"]["state_0_prob"],
        "state_1_prob": predictions["predictions"]["state_1_prob"]
    })
    save_dataframe(viterbi_df, str(viterbi_file))
    logger.info(f"Saved Viterbi path to {viterbi_file}")
    
    # Create run summary
    run_summary = {
        "timestamp": get_timestamp(),
        "config_file": "hmm_spx_studentt.yaml",
        "model_type": "Student-t HMM",
        "n_states": config["model"]["n_states"],
        "n_observations": len(predictions["predictions"]),
        "n_features": len(predictions["features"].columns),
        "files": {
            "params": str(params_file),
            "model": str(model_file),
            "posteriors": str(predictions_file),
            "features": str(features_file),
            "viterbi": str(viterbi_file)
        }
    }
    
    summary_file = Path(output_dir) / "last_run.json"
    save_json(run_summary, str(summary_file))
    logger.info(f"Saved run summary to {summary_file}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train S&P 500 Student-t HMM")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hmm_spx_studentt.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory for artifacts"
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
        logger.info("Starting HMM training pipeline...")
        
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Load and prepare data
        feature_data, feature_columns = load_and_prepare_data(config)
        
        # Train model
        model = train_hmm_model(feature_data, feature_columns, config)
        
        # Generate predictions
        predictions = generate_predictions(model, feature_data, feature_columns)
        
        # Save results
        save_results(model, predictions, config, args.output_dir)
        
        logger.info("HMM training pipeline completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Model: {config['model']['name']}")
        print(f"States: {config['model']['n_states']}")
        print(f"Features: {len(feature_columns)}")
        print(f"Observations: {len(feature_data)}")
        print(f"Output directory: {args.output_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
