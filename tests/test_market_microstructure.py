import pytest
import numpy as np
from core.market_microstructure import MicrostructureAnalyzer

@pytest.fixture
def microstructure():
    return MicrostructureAnalyzer(window_size=10)

def test_detect_spoofing(microstructure):
    order_book = {
        'bids': [[100.0, 2.0], [99.9, 1.0]],
        'asks': [[100.1, 2.0], [100.2, 1.0]]
    }
    assert not microstructure.detect_spoofing(order_book)

def test_calculate_liquidity_holes(microstructure):
    order_book = {
        'bids': [[100.0, 2.0], [99.8, 1.0]],
        'asks': [[100.1, 2.0], [100.3, 1.0]]
    }
    result = microstructure.calculate_liquidity_holes(order_book, levels=2)
    assert 'bid_holes' in result
    assert 'ask_holes' in result
    assert 'total_liquidity' in result
    assert 'avg_spread' in result

def test_update(microstructure):
    order_book = {
        'bids': [[100.0, 2.0]],
        'asks': [[100.1, 2.0]]
    }
    result = microstructure.update(price=100.05, order_book=order_book)
    assert 'spoofing_detected' in result
    assert 'liquidity_holes' in result
    assert 'market_regime' in result