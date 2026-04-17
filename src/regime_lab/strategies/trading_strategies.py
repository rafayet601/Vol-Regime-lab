"""
Complete Trading Strategies Module
===================================
Regime-Based Trading System for S&P 500

v2.0 — Uses the functional feature API (build_features, get_feature_cols).
"""

import logging
from typing import Dict

from ..data.features import build_features, get_feature_cols, load_spx_data, load_vix_data
from ..models.hmm_studentt import StudentTHMM

logger = logging.getLogger(__name__)


class RegimeBasedPositionSizer:
    """Adjust position sizes based on market regime."""

    def __init__(self, base_capital: float = 100000):
        self.base_capital = base_capital
        self.model = None

    def get_current_regime(self) -> Dict:
        """Get current market regime."""
        logger.info("Fetching current market regime...")

        # Load data using v2.0 API
        prices, returns = load_spx_data(start_date="2020-01-01")

        # Attempt to load VIX for Tier 2 features
        try:
            vix_df = load_vix_data(start_date="2020-01-01")
            tier = 2
        except Exception:
            vix_df = None
            tier = 1

        features = build_features(returns, prices=prices, vix_df=vix_df)
        feature_cols = get_feature_cols(features, tier=tier)

        X = features[feature_cols].values
        self.model = StudentTHMM(n_states=2, max_iter=50, random_seed=42)
        self.model.fit(X)

        current_state = self.model.predict(X[-1:])[0]
        current_probs = self.model.predict_proba(X[-1:])
        confidence = current_probs[0].max()

        return {
            'regime': current_state,
            'confidence': float(confidence),
            'is_high_vol': current_state == 1,  # State 1 = high-vol after label sorting
            'date': features.index[-1]
        }

    def calculate_position_size(self, target_allocation_pct: float = 100) -> Dict:
        """Calculate position size based on regime."""
        regime = self.get_current_regime()

        if regime['is_high_vol']:
            if regime['confidence'] > 0.8:
                allocation = target_allocation_pct * 0.5
                message = "🔴 HIGH VOL (high confidence): Reduce to 50%"
            else:
                allocation = target_allocation_pct * 0.7
                message = "🟡 HIGH VOL (medium confidence): Reduce to 70%"
        else:
            if regime['confidence'] > 0.8:
                allocation = target_allocation_pct * 1.0
                message = "🟢 LOW VOL (high confidence): Full allocation"
            else:
                allocation = target_allocation_pct * 0.9
                message = "🟢 LOW VOL (medium confidence): 90% allocation"

        position_size = self.base_capital * (allocation / 100)

        return {
            'position_size': float(position_size),
            'allocation_pct': float(allocation),
            'regime': 'HIGH_VOL' if regime['is_high_vol'] else 'LOW_VOL',
            'confidence': regime['confidence'],
            'message': message,
            'cash_reserve': float(self.base_capital - position_size),
            'date': regime['date']
        }


class RegimeAwareTradingSystem:
    """Complete trading system with regime detection."""

    def __init__(self, capital: float = 100000):
        self.capital = capital
        self.position_sizer = RegimeBasedPositionSizer(capital)

    def daily_routine(self):
        """Run this every morning before market open."""
        print("\n" + "="*70)
        print("🎯 REGIME-AWARE TRADING SYSTEM - DAILY ROUTINE")
        print("="*70)

        # Get position sizing
        position = self.position_sizer.calculate_position_size()

        print(f"\n📅 Date: {position['date'].date()}")
        print(f"\n1. CURRENT REGIME: {position['regime']}")
        print(f"   Confidence: {position['confidence']:.1%}")
        print(f"   {position['message']}")

        print(f"\n2. POSITION SIZING:")
        print(f"   Recommended Capital: ${position['position_size']:,.0f}")
        print(f"   Allocation:          {position['allocation_pct']:.0f}%")
        print(f"   Cash Reserve:        ${position['cash_reserve']:,.0f}")

        print(f"\n3. TRADING RECOMMENDATIONS:")
        if position['regime'] == 'LOW_VOL' and position['confidence'] > 0.7:
            print("   ✅ FAVORABLE CONDITIONS:")
            print("      • Take new long positions")
            print("      • Full position sizing")
            print("      • Normal stop losses (2%)")
            print("      • Expected Sharpe: 1.82")
        elif position['regime'] == 'HIGH_VOL' and position['confidence'] > 0.7:
            print("   ⚠️  UNFAVORABLE CONDITIONS:")
            print("      • Reduce exposure by 50%")
            print("      • Widen stop losses (3-4%)")
            print("      • Consider defensive positions")
            print("      • Wait for regime change")
        else:
            print("   🟡 UNCERTAIN CONDITIONS:")
            print("      • Maintain current positions")
            print("      • Wait for high confidence signal")
            print("      • No new positions")

        print("\n" + "="*70)

        return position


def main():
    """Run the trading system."""
    system = RegimeAwareTradingSystem(capital=100000)
    result = system.daily_routine()

    print("\n📊 DETAILED ANALYSIS:")
    print(f"   • Your capital is divided:")
    print(f"     - Equity exposure: ${result['position_size']:,.0f}")
    print(f"     - Cash/Safety:     ${result['cash_reserve']:,.0f}")

    print("\n💡 INTERPRETATION:")
    if result['regime'] == 'LOW_VOL':
        print("   Markets are calm. Good time for:")
        print("   • Long equity positions")
        print("   • Selling puts (collect premium)")
        print("   • Bull spreads")
    else:
        print("   Markets are volatile. Consider:")
        print("   • Reduce equity exposure")
        print("   • Increase cash/bonds")
        print("   • Buy protection (puts)")
        print("   • Wait for regime change")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
