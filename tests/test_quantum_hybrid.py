import numpy as np
from core.quantum_hybrid import QuantumHybrid


def test_predict_basic():
    model = QuantumHybrid()
    # simple training data
    X = np.zeros((10, model._input_dim))
    y = np.zeros(10)
    # train synchronously
    import asyncio
    asyncio.run(model.train(X, y, epochs=1))
    output = model.predict({'momentum': 0.0, 'volatility': 0.1, 'rsi': 55.0, 'obv': 0})
    assert 0.0 <= output <= 1.0

