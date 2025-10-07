# Regime Lab

A comprehensive framework for S&P 500 regime detection using 2-state Student-t Hidden Markov Models with Baum-Welch algorithm.

## Overview

Regime Lab implements a sophisticated approach to financial regime detection, combining:
- **2-state Hidden Markov Model** with Student-t emissions for fat-tail modeling
- **Baum-Welch algorithm** for parameter estimation
- **Rolling volatility features** for regime characterization
- **Comprehensive diagnostics** and visualization tools

## Features

- 🏛️ **S&P 500 Data Loading**: Automated data retrieval and caching via yfinance
- 📊 **Feature Engineering**: Rolling volatility, absolute returns, negative returns indicators, z-scores
- 🎯 **Student-t HMM**: 2-state model with diagonal covariance structure
- 🔄 **Baum-Welch Training**: Maximum likelihood parameter estimation
- 📈 **Regime Visualization**: Price overlays with regime probabilities
- 📋 **Diagnostics**: Duration statistics, persistence tests, transition analysis
- 🧪 **Testing**: Comprehensive test suite with >95% coverage
- 🛠️ **Dev Tools**: Pre-commit hooks, linting, formatting, Docker support

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd regime-lab

# Create virtual environment
make venv
source venv/bin/activate

# Install dependencies
make install
```

### Basic Usage

```bash
# Train the HMM model
make train
# or
regime-train --config configs/hmm_spx_studentt.yaml

# Generate regime plots
make plot
# or
regime-plot --artifacts-dir artifacts --output-dir reports/figures
```

### Programmatic Usage

```python
from regime_lab.data.loader import load_spx_data
from regime_lab.data.features import engineer_spx_features
from regime_lab.models.hmm_studentt import create_and_fit_hmm
from regime_lab.eval.diagnostics import quick_regime_summary

# Load and prepare data
price_data, returns_data = load_spx_data(start_date="2000-01-01")
feature_data = engineer_spx_features(returns_data, rolling_window=20)

# Train HMM
feature_columns = ['rolling_std', 'abs_returns', 'negative_returns']
X = feature_data[feature_columns].values
model = create_and_fit_hmm(X, n_states=2, df=5.0)

# Generate predictions
predicted_states = model.predict_states(X)
state_probabilities = model.predict_proba(X)

# Analyze results
results_df = pd.DataFrame({
    'date': feature_data.index,
    'predicted_state': predicted_states,
    'state_0_prob': state_probabilities[:, 0],
    'state_1_prob': state_probabilities[:, 1]
})
quick_regime_summary(results_df)
```

## Project Structure

```
regime-lab/
├── pyproject.toml                 # Dependencies and CLI entry points
├── Makefile                       # Development commands
├── configs/
│   └── hmm_spx_studentt.yaml     # Model configuration
├── src/regime_lab/
│   ├── data/
│   │   ├── loader.py             # S&P 500 data loading
│   │   └── features.py           # Feature engineering
│   ├── models/
│   │   ├── hmm_studentt.py       # Student-t HMM implementation
│   │   └── garch.py              # GARCH helper functions
│   ├── plotting/
│   │   └── regimes.py            # Visualization tools
│   ├── eval/
│   │   └── diagnostics.py        # Model diagnostics
│   └── utils/
│       ├── config.py             # Configuration utilities
│       └── io.py                 # I/O helpers
├── scripts/
│   ├── train_hmm.py              # Training script
│   └── plot_regimes.py           # Plotting script
├── tests/
│   ├── test_features.py          # Feature engineering tests
│   └── test_loader.py            # Data loading tests
├── artifacts/                    # Model outputs
└── reports/figures/              # Generated plots
```

## Model Architecture

### Hidden Markov Model

The Student-t HMM consists of:

- **States**: 2 regimes (low/high volatility)
- **Emissions**: Student-t distributions with diagonal covariance
- **Parameters**:
  - `π`: Initial state probabilities
  - `A`: Transition probability matrix
  - `μ`: State-specific mean vectors
  - `σ`: State-specific scale parameters (diagonal)
  - `ν`: Degrees of freedom (fixed at 5.0)

### Feature Engineering

- **Rolling Volatility**: 20-day rolling standard deviation (annualized)
- **Absolute Returns**: Magnitude of returns for volatility clustering
- **Negative Returns**: Binary indicator for downside risk
- **Z-Score Returns**: Standardized returns for regime normalization

### Baum-Welch Algorithm

1. **Initialization**: K-means clustering for parameter initialization
2. **E-Step**: Compute forward-backward probabilities
3. **M-Step**: Update parameters via maximum likelihood
4. **Convergence**: Check log-likelihood improvement

## Configuration

The model is configured via `configs/hmm_spx_studentt.yaml`:

```yaml
data:
  symbol: "^GSPC"
  start_date: "2000-01-01"
  end_date: "2024-01-01"

features:
  rolling_window: 20
  additional_features:
    - "abs_returns"
    - "negative_returns"
    - "z_score_returns"

model:
  n_states: 2
  emission_type: "student_t"

training:
  max_iterations: 100
  tolerance: 1e-6
  random_seed: 42
```

## Development

### Code Quality

```bash
# Format code
make format

# Lint code
make lint

# Run tests
make test
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/regime_lab --cov-report=html
```

### Docker Support

```bash
# Build Docker image
make docker-build

# Run in container
make docker-run
```

## Output Artifacts

After training, the following artifacts are generated in `artifacts/`:

- `params.json`: Model parameters and configuration
- `model.pkl`: Trained model object
- `posteriors.csv`: State probabilities and predictions
- `features.csv`: Engineered features
- `viterbi.csv`: Most likely state sequence
- `last_run.json`: Run summary and metadata

## Visualization

The plotting module generates several visualizations:

- **Price with Regimes**: S&P 500 price with regime overlays
- **Feature Analysis**: Feature distributions by regime
- **Volatility Comparison**: Rolling vs. GARCH volatility
- **Regime Transitions**: Transition probability matrix and sequence

## Diagnostics

Comprehensive model diagnostics include:

- **Duration Statistics**: Mean, median, min/max regime durations
- **Persistence Tests**: Memoryless property testing
- **Transition Analysis**: Transition probability matrix
- **Regime Characteristics**: Feature statistics by regime

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass and code is formatted
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{regime_lab,
  title={Regime Lab: S&P 500 Regime Detection with Student-t HMM},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/regime-lab}
}
```

## Acknowledgments

- Built with [pomegranate](https://github.com/jmschrei/pomegranate) for HMM implementation
- Data provided by [yfinance](https://github.com/ranaroussi/yfinance)
- GARCH models via [arch](https://github.com/bashtage/arch)
- Visualization with [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/)
