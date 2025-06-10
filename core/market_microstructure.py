import numpy as np
from scipy.stats import skew, kurtosis
from collections import deque
import warnings
warnings.filterwarnings('ignore')  # Disable scipy warnings

class MicrostructureAnalyzer:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.volume_history = deque(maxlen=window_size)
        self.price_history = deque(maxlen=window_size)
        self.spread_history = deque(maxlen=window_size//5)  # Track for dynamic thresholds
        self.order_count_history = deque(maxlen=window_size)  # For entropy correction
        
        # **Dynamic thresholding system**
        self.baseline_volatility = None
        self.min_tick_size = 0.01  # Auto-detect in live trading
        self.spoofing_threshold_multiplier = 2.5  # Starting point, adjusts automatically

        # Kalman Filter params - now with volatility scaling
        self.process_variance = 1e-4
        self.measurement_variance = 0.1**2  # Will adapt in update()
        self.posteri_estimate = 1.0
        self.posteri_error_estimate = 1.0

    def _kalman_update(self, measurement):
        """Adaptive thresholding with volatility scaling"""
        # **Auto-adjust measurement noise based on recent volatility**
        if len(self.price_history) > 10:
            recent_vol = np.std(np.diff(list(self.price_history)[-10:]))
            self.measurement_variance = max(0.01, recent_vol)**2

        # Prediction
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # Update
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate

    def _get_dynamic_threshold(self):
        """Adjusts thresholds based on market conditions"""
        if not self.baseline_volatility and len(self.price_history) >= 20:
            self.baseline_volatility = np.std(np.diff(list(self.price_history)))
        
        current_vol = np.std(np.diff(list(self.price_history)[-20:])) if len(self.price_history) >= 20 else 1.0
        vol_ratio = current_vol / self.baseline_volatility if self.baseline_volatility else 1.0
        
        # **Scale threshold by volatility and recent spread**
        avg_spread = np.mean(self.spread_history) if self.spread_history else 0.0
        spread_adj = avg_spread / (10 * self.min_tick_size) if self.min_tick_size > 0 else 1.0
        
        return self.spoofing_threshold_multiplier * vol_ratio * max(1.0, spread_adj**0.5)

    def detect_spoofing(self, order_book):
        bids, asks = np.array(order_book['bids']), np.array(order_book['asks'])

        if len(bids) == 0 or len(asks) == 0:
            return False

        # **1. Enhanced Volume Entropy with Order Count Penalty**
        bid_volumes = bids[:, 1]
        ask_volumes = asks[:, 1]
        bid_orders = len(bid_volumes)
        ask_orders = len(ask_volumes)
        
        # **Add order count stability term**
        bid_entropy = -np.sum(bid_volumes * np.log(bid_volumes + 1e-10)) * np.log1p(bid_orders)
        ask_entropy = -np.sum(ask_volumes * np.log(ask_volumes + 1e-10)) * np.log1p(ask_orders)
        entropy_diff = abs(bid_entropy - ask_entropy) / max(bid_entropy, ask_entropy, 1e-5)

        # 2. Price clustering with tick-size awareness
        bid_prices = bids[:, 0]
        ask_prices = asks[:, 0]
        bid_clusters = sum(abs(price % self.min_tick_size) < 0.1*self.min_tick_size for price in bid_prices)
        ask_clusters = sum(abs(price % self.min_tick_size) < 0.1*self.min_tick_size for price in ask_prices)
        clustering_score = (bid_clusters + ask_clusters) / (len(bids) + len(asks))

        # 3. Order size distribution (now with power-law check)
        all_volumes = np.concatenate([bid_volumes, ask_volumes])
        volume_skew = skew(all_volumes)
        volume_kurt = kurtosis(all_volumes)
        
        # **Power-law test for spoofing patterns**
        sorted_volumes = np.sort(all_volumes)[::-1]
        power_law_coeff = np.polyfit(np.log1p(np.arange(1, len(sorted_volumes)+1)), 
                                 np.log1p(sorted_volumes), 1)[0]
        power_law_score = abs(power_law_coeff + 1.0)  # -1 is perfect power law

        # **Dynamic scoring with market-context weights**
        anomaly_score = (
            entropy_diff * 0.4 +
            clustering_score * 0.3 +
            abs(volume_skew) * 0.15 +
            abs(volume_kurt - 3) * 0.1 +
            power_law_score * 0.05
        )

        dynamic_threshold = self._kalman_update(anomaly_score) * self._get_dynamic_threshold()

        # **Extreme event override**
        if anomaly_score > 5.0 or power_law_score < 0.2:
            return True

        return anomaly_score > dynamic_threshold

    def calculate_liquidity_holes(self, order_book, levels=10):
        bids, asks = order_book['bids'][:levels], order_book['asks'][:levels]

        if not bids or not asks:
            return {'bid_holes': [], 'ask_holes': []}

        # **Reference spread for relative gap sizing**
        current_spread = asks[0][0] - bids[0][0]
        self.spread_history.append(current_spread)
        avg_spread = np.mean(self.spread_history) if self.spread_history else current_spread

        # Bid side analysis - now uses % of avg spread
        bid_gaps = []
        for i in range(1, len(bids)):
            gap = bids[i-1][0] - bids[i][0]
            if gap > 0.5 * avg_spread:  # Now relative to market spread
                severity = (gap / avg_spread) * (1.0 / i)  # Weight by depth level
                bid_gaps.append((bids[i-1][0], bids[i][0], severity))

        # Ask side analysis
        ask_gaps = []
        for i in range(1, len(asks)):
            gap = asks[i][0] - asks[i-1][0]
            if gap > 0.5 * avg_spread:
                severity = (gap / avg_spread) * (1.0 / i)
                ask_gaps.append((asks[i-1][0], asks[i][0], severity))

        return {
            'bid_holes': sorted(bid_gaps, key=lambda x: -x[2]),  # Sort by severity
            'ask_holes': sorted(ask_gaps, key=lambda x: -x[2]),
            'total_liquidity': sum(b[1] for b in bids) + sum(a[1] for a in asks),
            'avg_spread': avg_spread
        }

    def estimate_market_regime(self):
        """Improved regime detection with confidence scoring"""
        if len(self.price_history) < 50:  # Increased minimum window
            return {"regime": "neutral", "confidence": 0.0}

        returns = np.diff(list(self.price_history))
        
        # **Multi-timeframe volatility comparison**
        short_window = min(20, len(returns)//2)
        std_short = np.std(returns[-short_window:])
        std_long = np.std(returns)
        
        if std_long == 0:
            return {"regime": "neutral", "confidence": 0.0}
            
        hurst = 0.5 + 0.5 * (np.log(std_short + 1e-10) - np.log(std_long + 1e-10)) / np.log(2)
        confidence = min(1.0, abs(hurst - 0.5) / 0.2)  # 0=neutral, 1=strong
        
        # **Add momentum confirmation**
        momentum = np.mean(returns[-5:]) / (std_long + 1e-10)
        
        if hurst > 0.6 and abs(momentum) > 0.5:
            regime = "trending"
        elif hurst < 0.4 and abs(momentum) < 0.3:
            regime = "mean_reverting"
        else:
            regime = "neutral"
            
        return {
            "regime": regime,
            "confidence": confidence,
            "hurst": hurst,
            "momentum": momentum
        }

    def update(self, price, order_book, timestamp=None):
        """Enhanced update with tick size detection"""
        if len(self.price_history) >= 1:
            # Auto-detect tick size from price changes
            last_price = self.price_history[-1]
            tick_candidate = round(abs(price - last_price), 8)
            if tick_candidate > 0:
                self.min_tick_size = tick_candidate if self.min_tick_size is None else min(self.min_tick_size, tick_candidate)
        
        self.price_history.append(price)
        self.order_count_history.append(len(order_book['bids']) + len(order_book['asks']))
        
        # Store spread for liquidity analysis
        if order_book['bids'] and order_book['asks']:
            self.spread_history.append(order_book['asks'][0][0] - order_book['bids'][0][0])

        spoofing_flag = self.detect_spoofing(order_book)
        holes = self.calculate_liquidity_holes(order_book)
        regime = self.estimate_market_regime()

        return {
            'spoofing_detected': spoofing_flag,
            'liquidity_holes': holes,
            'market_regime': regime,
            'current_tick_size': self.min_tick_size,
            'order_book_imbalance': len(order_book['bids']) / max(1, len(order_book['asks']))
        }