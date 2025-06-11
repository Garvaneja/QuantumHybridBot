import torch
from core.meta_lstm import EnhancedMetaLSTM


def test_detect_hft_patterns_basic():
    model = EnhancedMetaLSTM()
    order_flow = {
        'bid_ask_spread': 0.01,
        'order_imbalance': 1.0,
        'mid_price_velocity': 0.0,
        'cancel_rate': 0.1,
        'fill_rate': 0.9,
        'large_trade_ratio': 0.2,
        'order_size_entropy': 1.0,
        'price_clustering': 0.05,
        'volume_ratio': 1.0,
        'lifetime_entropy': 1.0,
    }
    output = model.detect_hft_patterns(order_flow)
    assert 0.0 <= output['hft_prob'] <= 1.0
    assert output['regime'] in {'TRENDING', 'MEAN_REVERTING', 'VOLATILE', 'UNCERTAIN'}
    assert 0.0 <= output['confidence'] <= 1.0

