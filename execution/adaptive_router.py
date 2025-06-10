import hashlib
from collections import deque
from typing import Dict, Optional
from core.nash_executor import NashExecutor

class AdaptiveRouter:
    def __init__(self):
        self.fingerprint_log = deque(maxlen=1000)
        self.brokers = ['kraken', 'binance', 'bybit']
        self.nash = NashExecutor()
        self.cooldown = False
        self.broker_latencies = {
            'kraken': {'rest': 120, 'ws': 30},
            'binance': {'rest': 80, 'ws': 20},
            'bybit': {'rest': 150, 'ws': 40}
        }
        self.broker_fees = {
            'kraken': 0.0026,
            'binance': 0.0010,
            'bybit': 0.0006
        }

    def prevalidate_order(self, order: Dict) -> Optional[Dict]:
        """Blocks trades during cooldown or invalid states"""
        if self.cooldown or not order.get('amount') or not order.get('symbol'):
            return None
        return order

    def _generate_fingerprint(self, order: Dict) -> str:
        """Deterministic order fingerprint for anti-front-running"""
        salt = f"{order.get('id', '')}{order.get('timestamp', '')}"
        return hashlib.sha256(
            f"{order['amount']}{order['symbol']}{salt}".encode()
        ).hexdigest()

    def _select_broker(self, order_type: str, symbol: str) -> str:
        """Optimal broker based on latency and fees"""
        if order_type == 'ioc':
            # Prioritize low-latency brokers for IOC
            return min(
                self.brokers,
                key=lambda b: self.broker_latencies[b]['ws']
            )
        else:
            # Prioritize low fees for iceberg orders
            return min(
                self.brokers,
                key=lambda b: self.broker_fees[b]
            )

    def _create_iceberg(self, order: Dict, order_book: Dict) -> Dict:
        """Iceberg order with liquidity-adjusted visibility"""
        visible = min(
            order['amount'] * 0.1,
            sum(v for _, v in order_book['asks' if order['side'] == 'buy' else 'bids'][:3]) * 0.05
        )
        return {
            **order,
            'params': {
                'hidden': True,
                'visible': max(visible, 0.001)  # Min lot size
            },
            'broker': self._select_broker('iceberg', order['symbol']),
            'fingerprint': self._generate_fingerprint(order)
        }

    def _create_ioc(self, order: Dict, order_book: Dict) -> Dict:
        """IOC order with slippage protection"""
        available = sum(
            v for _, v in order_book['asks' if order['side'] == 'buy' else 'bids'][:3]
        )
        order['amount'] = min(order['amount'], available * 0.1)  # Max 10% of top 3 levels
        return {
            **order,
            'params': {'timeInForce': 'IOC'},
            'broker': self._select_broker('ioc', order['symbol']),
            'fingerprint': self._generate_fingerprint(order)
        }

    def route_order(self, order: Dict, order_book: Dict) -> Dict:
        """
        Routes order based on:
        1. Nash equilibrium strategy
        2. Liquidity conditions
        3. Broker performance
        """
        if not self.prevalidate_order(order):
            raise ValueError("Invalid order or cooldown active")

        action = self.nash.optimal_response(order_book)['strategy']
        
        if action == 'iceberg':
            routed = self._create_iceberg(order, order_book)
        else:
            routed = self._create_ioc(order, order_book)

        self.fingerprint_log.append(routed['fingerprint'])
        return routed



