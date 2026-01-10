from abc import ABC, abstractmethod


class CostModel(ABC):
    @abstractmethod
    def calculate(self, price: float, shares: float) -> float:
        pass


class SimpleCostModel(CostModel):
    def __init__(self, spread_bps: float = 5, commission_bps: float = 1):
        self.spread_bps = spread_bps
        self.commission_bps = commission_bps

    def calculate(self, price: float, shares: float) -> float:
        notional = price * abs(shares)
        spread_cost = notional * self.spread_bps / 10000
        commission = notional * self.commission_bps / 10000
        return spread_cost + commission

    def __repr__(self):
        return f"SimpleCostModel(spread={self.spread_bps}bps, commission={self.commission_bps}bps)"


class TieredCostModel(CostModel):
    def __init__(self, base_spread_bps: float = 5, impact_bps_per_1000: float = 0.5):
        self.base_spread_bps = base_spread_bps
        self.impact_bps_per_1000 = impact_bps_per_1000

    def calculate(self, price: float, shares: float) -> float:
        notional = price * abs(shares)
        base_cost = notional * self.base_spread_bps / 10000
        # market impact scales with size
        impact = notional * (abs(shares) / 1000) * self.impact_bps_per_1000 / 10000
        return base_cost + impact


class ZeroCostModel(CostModel):
    def calculate(self, price: float, shares: float) -> float:
        return 0.0
