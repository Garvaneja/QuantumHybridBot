import pandas as pd
import numpy as np
import asyncio
import os
import sys

# Allow running without installing as a package
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.quantum_hybrid import QuantumHybrid


def simulate_ohlc(num=500, seed=42):
    """Generate a synthetic OHLCV data frame."""
    rng = np.random.default_rng(seed)
    price = 100 + rng.standard_normal(num).cumsum()
    volume = rng.uniform(50, 150, num)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=num, freq="H"),
        "open": price,
        "high": price + rng.random(num) * 0.5,
        "low": price - rng.random(num) * 0.5,
        "close": price + rng.standard_normal(num) * 0.1,
        "volume": volume,
    })
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["momentum"] = df["close"].diff(10)
    df["volatility"] = df["close"].rolling(20).std()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    return df.dropna()


def prepare_data(df: pd.DataFrame):
    features = df[["momentum", "volatility", "rsi", "obv"]].to_dict("records")
    labels = (df["close"].shift(-1) > df["close"]).astype(float).values
    return features[:-1], labels[:-1]


async def run_backtest():
    df = compute_indicators(simulate_ohlc())
    X, y = prepare_data(df)

    split = int(len(X) * 0.7)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = QuantumHybrid()
    await model.train(X_train, y_train, epochs=20)

    predictions = [model.predict(f) for f in X_test]
    signals = np.where(np.array(predictions) > 0.55, 1,
                       np.where(np.array(predictions) < 0.45, -1, 0))
    all_returns = df["close"].pct_change().shift(-1)
    returns = all_returns.iloc[split:split + len(signals)].fillna(0).values
    pnl = np.nansum(signals * returns)
    cumulative = np.cumprod(1 + signals * returns) - 1
    final_return = cumulative[-1]

    print("Final return: {:.2%}".format(final_return))
    print("Average PnL per trade: {:.4f}".format(pnl / len(returns)))


if __name__ == "__main__":
    asyncio.run(run_backtest())
