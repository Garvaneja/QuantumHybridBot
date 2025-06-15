import argparse
import asyncio
import ccxt
import pandas as pd
import numpy as np
from core.quantum_hybrid import QuantumHybrid


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['momentum'] = df['close'].diff(10)
    df['volatility'] = df['close'].rolling(20).std()
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return df.dropna()


def prepare_data(df: pd.DataFrame):
    feats = df[['momentum', 'volatility', 'rsi', 'obv']].to_dict('records')
    labels = (df['close'].shift(-1) > df['close']).astype(float).fillna(0).values
    return feats[:-1], labels[:-1]


async def run_backtest(symbol: str, exchange_name: str, timeframe: str, limit: int):
    exch_class = getattr(ccxt, exchange_name)
    exchange = exch_class()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    exchange.close()
    df = pd.DataFrame(ohlcv, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df = compute_features(df)
    X, y = prepare_data(df)

    split = int(len(X) * 0.7)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = QuantumHybrid()
    await model.train(X_train, y_train, epochs=20)

    cash = 0.0
    pos = 0.0
    last_price = None

    for i, feat in enumerate(X_test):
        price = df.iloc[split + i]['close']
        signal = model.predict(feat)
        if signal > 0.55 and pos <= 0:
            pos = 1
            cash -= price
        elif signal < 0.45 and pos >= 0:
            pos = -1
            cash += price
        last_price = price
    if pos != 0 and last_price is not None:
        cash += pos * last_price
    return cash


def main():
    parser = argparse.ArgumentParser(description="Simple backtest runner")
    parser.add_argument('--symbol', default='ETH/USD')
    parser.add_argument('--exchange', default='kraken')
    parser.add_argument('--timeframe', default='1h')
    parser.add_argument('--limit', type=int, default=200)
    args = parser.parse_args()
    pnl = asyncio.run(run_backtest(args.symbol, args.exchange, args.timeframe, args.limit))
    print(f"Backtest PnL: {pnl:.2f} {args.symbol.split('/')[-1]}")


if __name__ == '__main__':
    main()
