import sys
import types
import numpy as np
import asyncio

# Provide a lightweight talib stub so main imports cleanly
stub = types.SimpleNamespace(
    MOM=lambda arr, timeperiod=10: np.zeros_like(arr, dtype=float),
    RSI=lambda arr, timeperiod=14: np.full_like(arr, 50.0, dtype=float),
    OBV=lambda closes, volumes: np.zeros_like(closes, dtype=float),
)
sys.modules['talib'] = stub

from main import QuantumTrader

class DummyExchange:
    def __init__(self):
        self.markets = {'ETH/EUR': {'precision': {'amount': 4}}}
    async def fetch_balance(self):
        return {'free': {'EUR': 1000}}

class DummyMarket:
    def get_optimal_price(self):
        return 100.0

def test_calculate_position_size_dynamic_quote():
    trader = QuantumTrader.__new__(QuantumTrader)
    trader.symbol = 'ETH/EUR'
    trader.exchange = DummyExchange()
    trader.market = DummyMarket()
    size = asyncio.run(trader._calculate_position_size())
    assert abs(size - 0.2) < 1e-6
