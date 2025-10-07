#!/usr/bin/env python3
"""Run the regime-aware trading system."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from regime_lab.strategies.trading_strategies import RegimeAwareTradingSystem

if __name__ == "__main__":
    print("="*70)
    print("🚀 LAUNCHING REGIME-AWARE TRADING SYSTEM")
    print("="*70)
    
    # Create system with your capital
    system = RegimeAwareTradingSystem(capital=100000)
    
    # Run daily routine
    result = system.daily_routine()
    
    print("\n✅ Analysis complete!")
    print(f"📧 Recommendation: {result['message']}")
