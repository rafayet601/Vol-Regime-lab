"""Regime visualization with price and probability overlays."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle
from pandas import DataFrame

from ..utils.config import ensure_dir

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class RegimePlotter:
    """Plotter for regime detection results."""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """Initialize regime plotter.
        
        Args:
            figsize: Figure size (width, height)
        """
        self.figsize = figsize
        self.colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
        
    def plot_price_with_regimes(
        self,
        price_data: DataFrame,
        regime_data: DataFrame,
        price_column: str = "close",
        regime_column: str = "predicted_state",
        title: str = "S&P 500 Price with Regime Detection",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot price data with regime overlays.
        
        Args:
            price_data: DataFrame with price data
            regime_data: DataFrame with regime predictions
            price_column: Name of the price column
            regime_column: Name of the regime column
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[3, 1])
        
        # Plot price data
        ax1.plot(price_data.index, price_data[price_column], 
                color='black', linewidth=1, alpha=0.8, label='S&P 500 Price')
        
        # Add regime background colors
        self._add_regime_background(ax1, regime_data, regime_column)
        
        ax1.set_title(title, fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot regime probabilities
        if 'state_0_prob' in regime_data.columns and 'state_1_prob' in regime_data.columns:
            ax2.fill_between(regime_data.index, 0, regime_data['state_0_prob'], 
                           color=self.colors[0], alpha=0.7, label='State 0 (Low Vol)')
            ax2.fill_between(regime_data.index, regime_data['state_0_prob'], 1, 
                           color=self.colors[1], alpha=0.7, label='State 1 (High Vol)')
            ax2.set_ylabel('State Probability', fontsize=12)
            ax2.set_ylim(0, 1)
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
        else:
            # Plot regime indicators
            regime_values = regime_data[regime_column]
            ax2.plot(regime_data.index, regime_values, 
                    color=self.colors[1], linewidth=2, marker='o', markersize=3)
            ax2.set_ylabel('Regime State', fontsize=12)
            ax2.set_ylim(-0.1, 1.1)
            ax2.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Date', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            ensure_dir(str(Path(save_path).parent))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        
        return fig
    
    def _add_regime_background(
        self, 
        ax: plt.Axes, 
        regime_data: DataFrame, 
        regime_column: str
    ) -> None:
        """Add regime background colors to plot.
        
        Args:
            ax: Matplotlib axes
            regime_data: DataFrame with regime data
            regime_column: Name of the regime column
        """
        regime_values = regime_data[regime_column].values
        dates = regime_data.index
        
        # Find regime changes
        regime_changes = np.where(np.diff(regime_values) != 0)[0]
        
        # Add regime backgrounds
        current_regime = regime_values[0]
        start_date = dates[0]
        
        for change_idx in regime_changes:
            end_date = dates[change_idx]
            
            # Add background for current regime
            if current_regime == 1:  # High volatility regime
                ax.axvspan(start_date, end_date, alpha=0.2, color=self.colors[1])
            
            current_regime = regime_values[change_idx + 1]
            start_date = end_date
        
        # Add background for final regime
        if current_regime == 1:
            ax.axvspan(start_date, dates[-1], alpha=0.2, color=self.colors[1])
    
    def plot_feature_analysis(
        self,
        feature_data: DataFrame,
        regime_data: DataFrame,
        feature_columns: List[str],
        title: str = "Feature Analysis by Regime",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot feature distributions by regime.
        
        Args:
            feature_data: DataFrame with feature data
            regime_data: DataFrame with regime predictions
            feature_columns: List of feature column names
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        n_features = len(feature_columns)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_features == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Combine data for analysis
        combined_data = feature_data.copy()
        combined_data['regime'] = regime_data['predicted_state'].values
        
        for i, feature in enumerate(feature_columns):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot distributions by regime
            for regime in [0, 1]:
                regime_data_feature = combined_data[combined_data['regime'] == regime][feature]
                ax.hist(regime_data_feature, bins=50, alpha=0.6, 
                       color=self.colors[regime], label=f'Regime {regime}')
            
            ax.set_title(f'{feature}', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(feature_columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            ensure_dir(str(Path(save_path).parent))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature analysis plot to {save_path}")
        
        return fig
    
    def plot_volatility_comparison(
        self,
        data: DataFrame,
        title: str = "Volatility Comparison",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot comparison of different volatility measures.
        
        Args:
            data: DataFrame with volatility data
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[2, 1])
        
        # Plot volatility measures
        volatility_cols = [col for col in data.columns if 'vol' in col.lower() or 'std' in col.lower()]
        
        for i, col in enumerate(volatility_cols):
            if col in data.columns:
                axes[0].plot(data.index, data[col], 
                           color=self.colors[i % len(self.colors)], 
                           label=col, linewidth=1.5)
        
        axes[0].set_title(title, fontsize=16, fontweight='bold')
        axes[0].set_ylabel('Volatility', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot returns for context
        if 'returns' in data.columns:
            axes[1].plot(data.index, data['returns'], 
                        color='black', alpha=0.7, linewidth=0.8)
            axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1].set_ylabel('Returns', fontsize=12)
            axes[1].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Date', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            ensure_dir(str(Path(save_path).parent))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved volatility comparison plot to {save_path}")
        
        return fig
    
    def plot_regime_transitions(
        self,
        regime_data: DataFrame,
        title: str = "Regime Transition Analysis",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot regime transition patterns.
        
        Args:
            regime_data: DataFrame with regime predictions
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot regime sequence
        regime_values = regime_data['predicted_state']
        ax1.plot(regime_data.index, regime_values, 
                color=self.colors[1], linewidth=2, marker='o', markersize=3)
        ax1.set_title('Regime Sequence Over Time', fontweight='bold')
        ax1.set_ylabel('Regime State')
        ax1.set_xlabel('Date')
        ax1.set_ylim(-0.1, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Plot transition matrix
        transition_matrix = self._compute_transition_matrix(regime_values)
        im = ax2.imshow(transition_matrix, cmap='Blues', aspect='auto')
        
        # Add text annotations
        for i in range(transition_matrix.shape[0]):
            for j in range(transition_matrix.shape[1]):
                text = ax2.text(j, i, f'{transition_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        ax2.set_title('Transition Probability Matrix', fontweight='bold')
        ax2.set_xlabel('To State')
        ax2.set_ylabel('From State')
        ax2.set_xticks([0, 1])
        ax2.set_yticks([0, 1])
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            ensure_dir(str(Path(save_path).parent))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved regime transitions plot to {save_path}")
        
        return fig
    
    def _compute_transition_matrix(self, regime_sequence: pd.Series) -> np.ndarray:
        """Compute transition probability matrix.
        
        Args:
            regime_sequence: Series of regime states
            
        Returns:
            2x2 transition matrix
        """
        n_states = 2
        transition_counts = np.zeros((n_states, n_states))
        
        for i in range(len(regime_sequence) - 1):
            current_state = int(regime_sequence.iloc[i])
            next_state = int(regime_sequence.iloc[i + 1])
            transition_counts[current_state, next_state] += 1
        
        # Normalize to get probabilities
        row_sums = transition_counts.sum(axis=1)
        transition_matrix = transition_counts / row_sums[:, np.newaxis]
        
        return transition_matrix
    
    def create_summary_dashboard(
        self,
        price_data: DataFrame,
        feature_data: DataFrame,
        regime_data: DataFrame,
        output_dir: str = "reports/figures"
    ) -> None:
        """Create comprehensive summary dashboard.
        
        Args:
            price_data: DataFrame with price data
            feature_data: DataFrame with feature data
            regime_data: DataFrame with regime predictions
            output_dir: Output directory for plots
        """
        ensure_dir(output_dir)
        
        logger.info("Creating summary dashboard...")
        
        # 1. Price with regimes
        self.plot_price_with_regimes(
            price_data, regime_data,
            save_path=f"{output_dir}/price_with_regimes.png"
        )
        
        # 2. Feature analysis
        feature_columns = [col for col in feature_data.columns 
                          if col not in ['returns', 'date', 'datetime']]
        if feature_columns:
            self.plot_feature_analysis(
                feature_data, regime_data, feature_columns,
                save_path=f"{output_dir}/feature_analysis.png"
            )
        
        # 3. Volatility comparison
        volatility_data = pd.concat([feature_data, regime_data], axis=1)
        self.plot_volatility_comparison(
            volatility_data,
            save_path=f"{output_dir}/volatility_comparison.png"
        )
        
        # 4. Regime transitions
        self.plot_regime_transitions(
            regime_data,
            save_path=f"{output_dir}/regime_transitions.png"
        )
        
        logger.info(f"Summary dashboard created in {output_dir}")


# Convenience functions
def plot_regimes_from_artifacts(
    artifacts_dir: str = "artifacts",
    output_dir: str = "reports/figures"
) -> None:
    """Create regime plots from saved artifacts.
    
    Args:
        artifacts_dir: Directory containing model artifacts
        output_dir: Directory to save plots
    """
    from ..utils.io import load_dataframe, load_json
    
    # Load artifacts
    predictions = load_dataframe(f"{artifacts_dir}/posteriors.csv")
    features = load_dataframe(f"{artifacts_dir}/features.csv")
    
    # Create plotter and dashboard
    plotter = RegimePlotter()
    plotter.create_summary_dashboard(
        price_data=features,  # Using features as proxy for price data
        feature_data=features,
        regime_data=predictions,
        output_dir=output_dir
    )


def quick_regime_plot(
    regime_data: DataFrame,
    feature_data: Optional[DataFrame] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Quick regime visualization.
    
    Args:
        regime_data: DataFrame with regime predictions
        feature_data: Optional feature data for background
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    plotter = RegimePlotter()
    
    if feature_data is not None:
        return plotter.plot_price_with_regimes(
            feature_data, regime_data, 
            price_column=feature_data.columns[0],
            save_path=save_path
        )
    else:
        return plotter.plot_regime_transitions(regime_data, save_path=save_path)
