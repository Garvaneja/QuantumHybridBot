import numpy as np

class RiskException(Exception):
    """Custom exception for risk management violations"""
    pass

class QuantumRiskManager:
    def __init__(self):
        self.base_equity = 1000  # Starting capital
        self.current_balance = 1000
        self.position_limits = {'ETH': 0.2, 'BTC': 0.1}  # Base allocation %
        self.trade_log = []
        self.pnl_history = []
        self.equity_curve = [1000]
        self.current_volatility = 0.0
        self.max_drawdown_limit = -0.05  # -5%
        self.var_confidence = 0.95  # 95% VaR

    def calculate_var(self) -> float:
        """Classical Value-at-Risk using percentiles"""
        if len(self.pnl_history) < 30:
            return -0.05  # Conservative default
        try:
            return np.percentile(self.pnl_history, 100 * (1 - self.var_confidence))
        except Exception:
            return -0.05

    def check_order(self, order: dict, order_book: dict = None) -> bool:
        """Comprehensive pre-trade risk checks"""
        asset = order.get('symbol', '').split('/')[0]
        amount = abs(order.get('amount', 0))

        # 1. Circuit breaker check
        if self._trigger_circuit_breaker():
            raise RiskException("🚨 Volatility spike - trading suspended")

        # 2. Drawdown check
        current_dd = self.calculate_max_drawdown()
        if current_dd < self.max_drawdown_limit:
            raise RiskException(f"🛑 Drawdown limit breached: {current_dd:.2%}")

        # 3. Dynamic position sizing
        limit = self._get_dynamic_limit(asset)
        if amount > limit:
            raise RiskException(f"⚠️ {asset} position exceeds {limit:.1%} limit")

        # 4. Liquidity check (if order book provided)
        if order_book and not self._check_liquidity(order['symbol'], amount, order_book):
            raise RiskException("⚠️ Order exceeds 10% of top 5 order book levels")

        return True

    def _get_dynamic_limit(self, asset: str) -> float:
        """Volatility-adjusted position limit"""
        base_limit = self.position_limits.get(asset, 0.1)
        if self.current_volatility == 0:
            return base_limit
        vol_adjustment = min(2.0, 0.5 / self.current_volatility)
        return base_limit * vol_adjustment

    def _check_liquidity(self, symbol: str, amount: float, order_book: dict) -> bool:
        """Validates order size against market depth"""
        bids = order_book.get('bids', [])
        if not bids:
            return False
        top_5_depth = sum(v for _, v in bids[:5])
        return amount <= top_5_depth * 0.1

    def calculate_max_drawdown(self, window=30) -> float:
        """Rolling max drawdown calculation"""
        if len(self.equity_curve) < 2:
            return 0.0
        window = min(window, len(self.equity_curve))
        rolling_peak = np.maximum.accumulate(self.equity_curve[-window:])
        trough = np.min(self.equity_curve[-window:])
        return (trough - rolling_peak[-1]) / rolling_peak[-1]

    def _trigger_circuit_breaker(self) -> bool:
        """Checks for extreme volatility spikes"""
        if len(self.pnl_history) < 10:
            return False
        short_term_vol = np.std(self.pnl_history[-10:])
        long_term_vol = np.std(self.pnl_history[-100:]) if len(self.pnl_history) >= 100 else short_term_vol
        return short_term_vol > 3 * long_term_vol

    def record_trade(self, result: dict):
        """Updates risk metrics post-trade"""
        pnl = result.get('pnl', 0)
        self.current_balance += pnl
        self.pnl_history.append(pnl)
        self.equity_curve.append(self.current_balance)

        self.current_volatility = np.std(self.pnl_history[-30:]) if len(self.pnl_history) >= 30 else 0.0

        print(
            f"📊 Risk Update | Balance: ${self.current_balance:,.2f} | "
            f"Δ: {pnl:+.2f} | σ: {self.current_volatility:.4f} | "
            f"VaR: {self.calculate_var():.2%} | DD: {self.calculate_max_drawdown():.2%}"
        )
