import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from costs import SimpleCostModel, ZeroCostModel, TieredCostModel


class TestSimpleCostModel:
    def test_linear_cost(self):
        model = SimpleCostModel(spread_bps=10, commission_bps=0)

        # 100 shares at $50 = $5000 notional
        # 10 bps = 0.1% = $5
        cost = model.calculate(price=50, shares=100)
        assert abs(cost - 5.0) < 0.01

    def test_spread_plus_commission(self):
        model = SimpleCostModel(spread_bps=5, commission_bps=5)

        # $10000 notional, 10 bps total = $10
        cost = model.calculate(price=100, shares=100)
        assert abs(cost - 10.0) < 0.01

    def test_zero_shares_zero_cost(self):
        model = SimpleCostModel(spread_bps=10, commission_bps=5)
        cost = model.calculate(price=100, shares=0)
        assert cost == 0

    def test_negative_shares_uses_abs(self):
        model = SimpleCostModel(spread_bps=10, commission_bps=0)

        cost_buy = model.calculate(price=50, shares=100)
        cost_sell = model.calculate(price=50, shares=-100)

        assert cost_buy == cost_sell


class TestZeroCostModel:
    def test_always_zero(self):
        model = ZeroCostModel()

        assert model.calculate(100, 1000) == 0
        assert model.calculate(50, 500) == 0
        assert model.calculate(0, 0) == 0


class TestTieredCostModel:
    def test_base_cost(self):
        model = TieredCostModel(base_spread_bps=10, impact_bps_per_1000=0)

        cost = model.calculate(price=100, shares=100)
        # $10000 * 10bps = $10
        assert abs(cost - 10.0) < 0.01

    def test_impact_increases_with_size(self):
        model = TieredCostModel(base_spread_bps=10, impact_bps_per_1000=1)

        small_cost = model.calculate(price=100, shares=100)
        large_cost = model.calculate(price=100, shares=1000)

        # larger trade should cost more per dollar
        small_per_dollar = small_cost / (100 * 100)
        large_per_dollar = large_cost / (100 * 1000)

        assert large_per_dollar > small_per_dollar
