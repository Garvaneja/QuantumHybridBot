import numpy as np
from nashpy import Game
from collections import deque
import pandas as pd

class NashExecutor:
    def __init__(self, latency_monitor=None):
        # Dynamic opponent models (updated in real-time)
        self.opponent_models = {
            'hedge_fund': {
                'spoof_prob': 0.5,  # Initial estimate
                'cancel_speed': 1.0,  # Relative to our latency
                'aggression': 0.5,
                'n_obs': 0  # Tracking observations
            },
            'market_maker': {
                'fade_prob': 0.6,
                'spread_elasticity': 0.3,
                'n_obs': 0
            }
        }
        
        # Strategy performance tracking
        self.strategy_history = {
            'iceberg': {'success': 0, 'fail': 0},
            'sniping': {'success': 0, 'fail': 0}
        }
        
        # Market state memory
        self.spread_history = deque(maxlen=1000)
        self.spoof_history = deque(maxlen=500)
        
        # Execution parameters
        self.latency_monitor = latency_monitor
        self.min_iceberg_size = 5.0  # BTC/contracts/etc
        self.max_snipe_risk = 0.02  # 2% of portfolio
        
        # Bayesian hyperparameters
        self.prior_strength = 10  # How quickly to adapt to new data

    def dynamic_payoff_matrix(self, order_book: dict, portfolio_risk: float) -> np.ndarray:
        """
        Data-driven payoff matrix based on:
        1) Historical strategy performance
        2) Current market liquidity
        3) Portfolio risk constraints
        """
        try:
            # Liquidity analysis
            bid_depth = sum(v for _, v in order_book.get('bids', [])[:5])
            ask_depth = sum(v for _, v in order_book.get('asks', [])[:5])
            spread = order_book['asks'][0][0] - order_book['bids'][0][0] if order_book.get('asks') and order_book.get('bids') else 0.0
            self.spread_history.append(spread)
            
            # Spoofing detection
            spoof_score = self._calc_spoof_score(order_book)
            self.spoof_history.append(spoof_score)
            
            # Strategy success rates (with Bayesian smoothing)
            iceberg_success = (self.strategy_history['iceberg']['success'] + 1) / \
                            (self.strategy_history['iceberg']['success'] + self.strategy_history['iceberg']['fail'] + 2)
            snipe_success = (self.strategy_history['sniping']['success'] + 1) / \
                          (self.strategy_history['sniping']['success'] + self.strategy_history['sniping']['fail'] + 2)
            
            # Depth-adjusted payoffs
            liquidity_ratio = min(bid_depth / ask_depth, 3.0) if ask_depth > 0 else 1.0
            iceberg_payoff = iceberg_success * (1 - spoof_score) * liquidity_ratio
            snipe_payoff = snipe_success * (spread / np.percentile(self.spread_history, 90)) if len(self.spread_history) > 10 else 0.5
            
            # Risk constraints
            iceberg_risk = min(portfolio_risk * 0.5, self.min_iceberg_size)
            snipe_risk = min(portfolio_risk * 0.2, self.max_snipe_risk)
            
            return np.array([
                [iceberg_payoff * iceberg_risk, -snipe_payoff * 0.7],  # Iceberg fail -> opponent snipes
                [snipe_payoff * snipe_risk * 1.3, -iceberg_payoff * 0.4]  # Snipe fail -> opponent icebergs
            ])
            
        except Exception as e:
            print(f"⚠️ Payoff matrix error: {e}")
            return np.array([[0.3, -0.1], [0.2, -0.2]])

    def optimal_response(self, order_book: dict, portfolio_risk: float) -> dict:
        """
        Nash Equilibrium strategy selection with:
        - Latency checks
        - Confidence scoring
        - Fallback logic
        """
        try:
            # Latency override
            if self.latency_monitor and self.latency_monitor.current_latency > 50:
                return {
                    'strategy': 'iceberg',
                    'confidence': 0.9,
                    'reason': 'high_latency'
                }
                
            payoff = self.dynamic_payoff_matrix(order_book, portfolio_risk)
            game = Game(payoff)
            
            # Find all Nash equilibria
            equilibria = list(game.support_enumeration())
            if not equilibria:
                return self._fallback_strategy(order_book)
                
            # Select most probable equilibrium
            strategies, _ = zip(*equilibria)
            avg_strategy = np.mean(strategies, axis=0)
            
            if np.isnan(avg_strategy).any():
                return self._fallback_strategy(order_book)
                
            # Confidence calculation
            dominance = abs(avg_strategy[0] - avg_strategy[1])
            confidence = min(0.95, dominance * 2)  # Scale to [0, 0.95]
            
            strategy = 'iceberg' if avg_strategy[0] >= avg_strategy[1] else 'sniping'
            
            # Update opponent models
            self._update_opponent_models(strategy, order_book)
            
            return {
                'strategy': strategy,
                'confidence': confidence,
                'payoff_matrix': payoff,
                'reason': 'nash_equilibrium'
            }
            
        except Exception as e:
            print(f"❌ Nash solver failed: {e}")
            return self._fallback_strategy(order_book)

    def _update_opponent_models(self, our_strategy: str, order_book: dict):
        """Bayesian updating of opponent behavior models"""
        bids, asks = order_book.get('bids', []), order_book.get('asks', [])
        if not bids or not asks:
            return
            
        # Detect opponent reactions
        spread = asks[0][0] - bids[0][0]
        mid_price = (asks[0][0] + bids[0][0]) / 2
        
        # Track market maker fading
        if our_strategy == 'iceberg' and len(asks) > 1:
            fade_distance = asks[1][0] - mid_price
            is_fade = fade_distance > spread * 1.5
            self.opponent_models['market_maker']['n_obs'] += 1
            n = self.opponent_models['market_maker']['n_obs']
            current_prob = self.opponent_models['market_maker']['fade_prob']
            new_prob = (current_prob * (n - 1) + is_fade) / n
            self.opponent_models['market_maker']['fade_prob'] = new_prob
            
        # Track hedge fund spoofing
        spoof_score = self._calc_spoof_score(order_book)
        self.opponent_models['hedge_fund']['n_obs'] += 1
        n = self.opponent_models['hedge_fund']['n_obs']
        current_spoof = self.opponent_models['hedge_fund']['spoof_prob']
        new_spoof = (current_spoof * (n - 1) + (spoof_score > 0.7)) / n
        self.opponent_models['hedge_fund']['spoof_prob'] = new_spoof

    def _calc_spoof_score(self, order_book: dict) -> float:
        """Quantifies spoofing likelihood (0-1)"""
        bids, asks = np.array(order_book.get('bids', [])), np.array(order_book.get('asks', []))
        if len(bids) == 0 or len(asks) == 0:
            return 0.0
            
        # Volume entropy asymmetry
        bid_volumes = bids[:, 1]
        ask_volumes = asks[:, 1]
        bid_entropy = -np.sum(bid_volumes * np.log(bid_volumes + 1e-10))
        ask_entropy = -np.sum(ask_volumes * np.log(ask_volumes + 1e-10))
        entropy_diff = abs(bid_entropy - ask_entropy) / max(bid_entropy, ask_entropy, 1e-5)
        
        # Cancellation patterns (if tracking order flow)
        cancel_rate = 0.0
        if hasattr(self, 'cancel_tracker'):
            cancel_rate = self.cancel_tracker.get_cancel_rate()
            
        # Composite score
        return min(1.0, entropy_diff * 0.6 + cancel_rate * 0.4)

    def _fallback_strategy(self, order_book: dict) -> dict:
        """Fallback when Nash solver fails"""
        spread = order_book['asks'][0][0] - order_book['bids'][0][0] if order_book.get('asks') and order_book.get('bids') else 0.0
        if spread < np.percentile(self.spread_history, 30) if len(self.spread_history) > 10 else 0.0005:
            return {'strategy': 'sniping', 'confidence': 0.7, 'reason': 'tight_spread'}
        else:
            return {'strategy': 'iceberg', 'confidence': 0.8, 'reason': 'fallback'}

    def log_strategy_outcome(self, strategy: str, success: bool):
        """Update strategy performance tracking"""
        key = 'success' if success else 'fail'
        self.strategy_history[strategy][key] += 1
        
        # Cap history to prevent stale data
        total = self.strategy_history[strategy]['success'] + self.strategy_history[strategy]['fail']
        if total > 1000:
            decay = 0.9
            self.strategy_history[strategy]['success'] = int(self.strategy_history[strategy]['success'] * decay)
            self.strategy_history[strategy]['fail'] = int(self.strategy_history[strategy]['fail'] * decay)

# Example usage:
if __name__ == "__main__":
    executor = NashExecutor()
    mock_book = {
        'bids': [[100.0, 2.5], [99.9, 3.1], [99.8, 1.7]],
        'asks': [[100.1, 1.8], [100.2, 2.3], [100.3, 0.9]]
    }
    decision = executor.optimal_response(mock_book, portfolio_risk=0.05)
    print(decision)


