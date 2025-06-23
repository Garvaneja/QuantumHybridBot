import argparse
import asyncio
import pandas as pd
import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from core.quantum_hybrid import QuantumHybrid
from core.risk_manager import QuantumRiskManager


def simulate_ohlc(num=500, seed=42):
    rng = np.random.default_rng(seed)
    price = 100 + rng.standard_normal(num).cumsum()
    volume = rng.uniform(50, 150, num)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2021-01-01', periods=num, freq='H'),
        'open': price,
        'high': price + rng.random(num) * 0.5,
        'low': price - rng.random(num) * 0.5,
        'close': price + rng.standard_normal(num) * 0.1,
        'volume': volume,
    })
    return df


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


def prepare_data(df: pd.DataFrame):
    features = df[['momentum', 'volatility', 'rsi', 'obv']].to_dict('records')
    labels = (df['close'].shift(-1) > df['close']).astype(float).values
    return features[:-1], labels[:-1]


class PaperTrader:
    def __init__(self, balance: float = 1000.0, risk: float = 0.02):
        self.initial_balance = float(balance)
        self.balance = float(balance)
        self.position = 0.0
        self.risk = float(risk)
        self.risk_manager = QuantumRiskManager(base_equity=self.balance)
        self.model = QuantumHybrid()

    async def train(self, X, y, epochs: int = 15):
        await self.model.train(X, y, epochs=epochs)

    def trade_step(self, feature: dict, price: float):
        signal = self.model.predict(feature)
        side = 'buy' if signal > 0.55 else ('sell' if signal < 0.45 else None)
        if side is None:
            return
        amount = self.balance * self.risk / price
        order = {'symbol': 'SIM/USD', 'side': side, 'amount': amount, 'price': price}
        try:
            self.risk_manager.check_order(order)
        except Exception as e:
            print(f'Risk reject: {e}')
            return
        if side == 'buy' and self.balance >= amount * price:
            self.balance -= amount * price
            self.position += amount
            print(f'BUY  {amount:.4f} @ {price:.2f}')
        elif side == 'sell':
            self.balance += amount * price
            self.position -= amount
            action = 'SELL' if self.position >= 0 else 'SHORT'
            print(f'{action} {amount:.4f} @ {price:.2f}')

    def finalize(self, last_price: float):
        self.balance += self.position * last_price
        pnl = self.balance - self.initial_balance
        self.risk_manager.record_trade({'pnl': pnl})
        print(f'Final balance: ${self.balance:.2f} | PnL: {pnl:+.2f}')
        return pnl


async def run_paper_trading(balance=1000.0, risk=0.02, seed=42, steps=400):
    df = compute_indicators(simulate_ohlc(num=steps + 100, seed=seed))
    X, y = prepare_data(df)
    split = int(len(X) * 0.7)
    trader = PaperTrader(balance=balance, risk=risk)
    await trader.train(X[:split], y[:split], epochs=20)
    prices = df['close'].iloc[split:-1].values
    for feat, price in zip(X[split:], prices):
        trader.trade_step(feat, price)
    trader.finalize(prices[-1])


def main():
    parser = argparse.ArgumentParser(description='Run paper trading simulation')
    parser.add_argument('--balance', type=float, default=1000.0, help='Starting capital')
    parser.add_argument('--risk', type=float, default=0.02, help='Risk per trade (fraction)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--steps', type=int, default=400, help='Number of data points')
    args = parser.parse_args()
    asyncio.run(run_paper_trading(balance=args.balance, risk=args.risk, seed=args.seed, steps=args.steps))


if __name__ == '__main__':
    main()
