import argparse
import asyncio
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ccxt

from core.quantum_hybrid import QuantumHybrid


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
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


def prepare_training_data(df: pd.DataFrame):
    features = df[['momentum', 'volatility', 'rsi', 'obv']].to_dict('records')
    labels = (df['close'].shift(-1) > df['close']).astype(float).fillna(0).values
    return features[:-1], labels[:-1]


async def generate_signals(symbol: str, exchange_name: str, timeframe: str, limit: int):
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class()
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    exchange.close()

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = compute_indicators(df)

    X, y = prepare_training_data(df)
    model = QuantumHybrid()
    await model.train(X, y, epochs=20)

    df = df.iloc[:-1].copy()
    df['signal'] = [model.predict(f) for f in X]
    return df


def plot_signals(df: pd.DataFrame, symbol: str, output: str = None):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['timestamp'], df['close'], color='tab:blue', label='Price')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(df['timestamp'], df['signal'], color='tab:orange', label='Signal')
    ax2.set_ylabel('Signal', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.suptitle(f'{symbol} Price vs Signal')
    fig.tight_layout()

    if output:
        plt.savefig(output)
    else:
        plt.show()


async def main():
    parser = argparse.ArgumentParser(description='Visualize QuantumHybrid signals versus price.')
    parser.add_argument('--symbol', default='ETH/USD', help='Trading pair symbol')
    parser.add_argument('--exchange', default='kraken', help='Exchange name for ccxt')
    parser.add_argument('--timeframe', default='1h', help='OHLCV timeframe')
    parser.add_argument('--limit', type=int, default=200, help='Number of bars to fetch')
    parser.add_argument('--output', help='Optional output PNG file path')
    args = parser.parse_args()

    df = await generate_signals(args.symbol, args.exchange, args.timeframe, args.limit)
    plot_signals(df, args.symbol, args.output)


if __name__ == '__main__':
    asyncio.run(main())
