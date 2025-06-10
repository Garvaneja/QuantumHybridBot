import numpy as np
import hashlib
from typing import Dict, Optional
from core.nash_executor import NashExecutor

class StealthRouter:
    def __init__(self):
        self.nash = NashExecutor()
        self.fingerprint_log = []
        self.brokers = ['kraken', 'binance', 'bybit']
        self.broker_stats = {
            'kraken': {'fee': 0.0026, 'latency': 120},
            'binance': {'fee': 0.0010, 'latency': 80},
            'bybit': {'fee': 0.0006, 'latency': 150}
        }

    def _generate_fingerprint(self, order: Dict) -> str:
        """Anti-front-running signature using order metadata"""
        salt = f"{order.get('id', '')}{order.get('timestamp', '')}"
        return hashlib.sha256(
            f"{order['amount']}{order['symbol']}{salt}".encode()
        ).hexdigest()

    def _select_broker(self, order_type: str) -> str:
        """Optimal broker for order type (IOC → low latency, Iceberg → low fee)"""
        return min(
            self.brokers,
            key=lambda x: self.broker_stats[x]['latency' if order_type == 'ioc' else 'fee']
        )

    def _validate_order_size(self, order: Dict, order_book: Dict) -> float:
        """Prevents slippage by capping order size"""
        side_key = 'asks' if order['side'] == 'buy' else 'bids'
        available = sum(v for _, v in order_book[side_key][:3])
        return min(order['amount'], available * 0.1)

    def _create_iceberg(self, order: Dict, order_book: Dict) -> Dict:
        """Liquidity-adaptive iceberg order"""
        side_key = 'asks' if order['side'] == 'buy' else 'bids'
        visible = min(
            order['amount'] * 0.1,
            sum(v for _, v in order_book[side_key][:3]) * 0.05
        )
        return {
            **order,
            'broker': self._select_broker('iceberg'),
            'fingerprint': self._generate_fingerprint(order),
            'params': {
                'hidden': True,
                'visible': round(max(visible, 0.001), 8)  # Enforce min lot size
            }
        }

    def _create_ioc(self, order: Dict, order_book: Dict) -> Dict:
        """Slippage-controlled IOC order"""
        return {
            **order,
            'amount': self._validate_order_size(order, order_book),
            'broker': self._select_broker('ioc'),
            'fingerprint': self._generate_fingerprint(order),
            'params': {'timeInForce': 'IOC'}
        }

    def route_order(self, order: Dict, order_book: Dict) -> Dict:
        """
        Routes order with:
        - Nash equilibrium strategy
        - Liquidity checks
        - Anti-detection measures
        """
        if not order.get('amount') or not order.get('symbol'):
            raise ValueError("Order missing 'amount' or 'symbol'")

        action = self.nash.optimal_response(order_book)['strategy']
        order['amount'] = round(order['amount'] * np.random.uniform(0.95, 1.05), 8)

        if action == 'iceberg':
            return self._create_iceberg(order, order_book)
        return self._create_ioc(order, order_book)

