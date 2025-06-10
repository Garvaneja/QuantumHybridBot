#!/usr/bin/env python3
import sys
import asyncio

# On Windows, force selector loop so aiodns works
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import time
import numpy as np
import talib
import ccxt.pro as ccxt
from typing import Dict, Optional
import traceback
from datetime import datetime
import os
from dotenv import load_dotenv

# Local imports
from core.quantum_hybrid import QuantumHybrid
from core.meta_lstm import EnhancedMetaLSTM
from core.risk_manager import QuantumRiskManager
from core.market_microstructure import MicrostructureAnalyzer
from execution.stealth_router import StealthRouter
from core.market_connector import QuantumFeed

# Load environment variables
load_dotenv()

class QuantumTrader:
    def __init__(self, symbol: str = 'ETH/USD'):
        self.symbol = symbol
        self._init_failed = False
        self._consecutive_errors = 0
        self.max_errors = 5
        self._init_components()
        self._init_state()

    def _init_components(self):
        """Initialize all components with error handling"""
        try:
            self.brain = QuantumHybrid()
            self.hft_detector = EnhancedMetaLSTM()
            self.router = StealthRouter()
            self.risk = QuantumRiskManager()
            self.microstructure = MicrostructureAnalyzer()
            self.market = QuantumFeed(exchange='kraken', symbol=self.symbol)
            
            # Initialize exchange
            self.exchange = ccxt.kraken({
                'apiKey': os.getenv('KRAKEN_API_KEY'),
                'secret': os.getenv('KRAKEN_API_SECRET'),
                'enableRateLimit': True,
                'timeout': 30000,
                'options': {
                    'defaultType': 'spot',
                    'adjustForTimeDifference': True
                }
            })
        except Exception as e:
            self._init_failed = True
            print(f"🔥 Critical initialization failed: {e}")
            traceback.print_exc()

    def _init_state(self):
        """Initialize trading state variables"""
        self.last_signal = None
        self.last_fetch_time = 0
        self.cached_ohlcv = {
            'momentum': 0.0,
            'volatility': 0.0,
            'rsi': 50.0,
            'obv': 0.0
        }
        self.last_price = None
        self._shutdown_flag = False

    async def start(self):
        """Start the trading system: train the model, spin up feeds, then trade."""
        if self._init_failed:
            raise RuntimeError("Cannot start - initialization failed")

        # 1) LOAD & TRAIN
        print("🔄 Loading historical data and training model…")
        try:
            # fetch 5m bars for training
            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, '5m', limit=500)
            closes = np.array([row[4] for row in ohlcv])
            # simple feature: last 10 closes → label: next-close-up?
            X, y = [], []
            for i in range(10, len(closes) - 1):
                X.append(closes[i-10:i])
                y.append(1.0 if closes[i+1] > closes[i] else 0.0)
            X_train = np.array(X)
            y_train = np.array(y)
            # train your QuantumHybrid
            await self.brain.train(X_train, y_train)
            print("✅ Model trained")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise

        try:
            # 2) Live startup
            await self.exchange.load_markets()
            await self.market.start()

            # wait up to 10s for first book
            for _ in range(20):
                book = await self.market.get_order_book()
                if book['bids'] and book['asks']:
                    print("✅ First book snapshot received")
                    break
                await asyncio.sleep(0.5)
            else:
                print("❗ Warning: no WS snapshot in 10s; will fallback to REST each cycle")

            print("✅ Trading system started successfully")
            await self._main_loop()

        finally:
            # 3) always clean up
            try:
                await self.exchange.close()
            except Exception as e:
                print(f"⚠️ Error closing exchange session: {e}")


    async def _main_loop(self):
        """Main trading loop with error handling"""
        while not self._shutdown_flag:
            cycle_start = time.time()
            try:
                await self.trade_cycle()
                self._consecutive_errors = 0
            except Exception as e:
                self._consecutive_errors += 1
                print(f"⚠️ Cycle error ({self._consecutive_errors}/{self.max_errors}): {e}")
                traceback.print_exc()
                if self._consecutive_errors >= self.max_errors:
                    print("🛑 Max errors reached - shutting down")
                    await self.shutdown()
                    break
            cycle_time = time.time() - cycle_start
            await asyncio.sleep(max(60 - cycle_time, 1))

    async def trade_cycle(self):
        """Single trading cycle execution"""
        if not await self._health_check():
            return

        price_data = await self.get_price_features()
        order_book = await self.market.get_order_book()
        if not order_book['bids'] or not order_book['asks']:
            print("🛠️ WS book empty — fetching REST snapshot")
            rest = await self.exchange.fetch_order_book(self.symbol, self.market.depth)
            order_book = {
                'bids': rest['bids'][: self.market.depth],
                'asks': rest['asks'][: self.market.depth],
            }

        signal = await self._generate_signal(price_data, order_book)
        if signal is None:
            return

        await self._execute_trade(signal, order_book)

    async def _health_check(self) -> bool:
        if not self.market.is_connected():
            print("🔌 Reconnecting market feed...")
            await self.market.start()
            return False
        try:
            balance = await self.exchange.fetch_balance()
            quote = self.symbol.split('/')[-1]
            available = None
            for code, amt in balance.get('free', {}).items():
                if code.upper().endswith(quote.upper()):
                    available = float(amt)
                    break
            if available is None or available < 10:
                print("💸 Insufficient balance or missing quote")
                return False
        except Exception as e:
            print(f"⚠️ Health check failed: {e}")
            return False
        return True

    async def _generate_signal(self, price_data: Dict, order_book: Dict) -> Optional[float]:
        if not order_book['bids'] or not order_book['asks']:
            print("📊 Empty order book - skipping")
            return None
        mid = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2
        self.last_price = mid
        signal = self.brain.predict(price_data, await self.get_sentiment())
        hft_features = self._extract_hft_features(order_book)
        hft_output = self.hft_detector.detect_hft_patterns(hft_features)
        spoof_flag = self.microstructure.detect_spoofing(order_book)

        print(f"{datetime.now().isoformat()}  📈 Signal: {signal:.2f}  📊 RSI: {price_data['rsi']:.1f}  ⚡ HFT: {hft_output['hft_prob']:.2f}  🎯 Spoof: {'✅' if spoof_flag else '❌'}")

        if hft_output['hft_prob'] < 0.3:
            print("⛔️ Low HFT probability")
            return None
        if abs(price_data['rsi'] - 50) < 5:
            print("⛔️ Neutral RSI")
            return None
        if self.last_signal and abs(self.last_signal - signal) < 0.1:
            print("⛔️ Similar to last signal")
            return None

        self.last_signal = signal
        return signal

    async def _execute_trade(self, signal: float, order_book: Dict):
        side = 'buy' if signal > 0.55 else ('sell' if signal < 0.45 else None)
        if not side:
            print("🤷 Neutral signal")
            return
        optimal_price = self.market.get_optimal_price()
        if optimal_price <= 0:
            print("💢 Invalid price - skipping")
            return
        order = {
            'symbol': self.symbol,
            'side': side,
            'amount': await self._calculate_position_size(),
            'price': round(optimal_price, 2),
            'timestamp': int(time.time())
        }
        routed = self.router.route_order(order, order_book)
        if not routed:
            print("🧊 Router rejected order")
            return
        try:
            print(f"⚡ Executing {side.upper()} order...")
            result = await self.exchange.create_order(
                symbol=routed['symbol'],
                type='limit',
                side=routed['side'],
                amount=routed['amount'],
                price=routed['price'],
                params=routed.get('params', {})
            )
            print(f"✅ Executed: {result['id']}")
            self.risk.record_trade(result)
        except Exception as e:
            print(f"❌ Execution failed: {e}")
            raise

    async def _calculate_position_size(self) -> float:
        try:
            balance = await self.exchange.fetch_balance()
            usd_balance = float(balance['free']['USD'])
            price = self.market.get_optimal_price()
            if price <= 0:
                return 0.0
            risk_per_trade = 0.02
            size = usd_balance * risk_per_trade / price
            return round(size, self.exchange.markets[self.symbol]['precision']['amount'])
        except Exception as e:
            print(f"💢 Position sizing failed: {e}")
            return 0.0

    async def get_price_features(self) -> Dict:
        try:
            if time.time() - self.last_fetch_time < 15 and self.cached_ohlcv:
                return self.cached_ohlcv
            ohlcv = await self.exchange.fetch_ohlcv(self.symbol, '5m', limit=50)
            closes = np.array([x[4] for x in ohlcv])
            volumes = np.array([x[5] for x in ohlcv])
            features = {
                'momentum': float(talib.MOM(closes, timeperiod=10)[-1]),
                'volatility': float(np.std(closes[-20:])),
                'rsi': float(talib.RSI(closes, timeperiod=14)[-1]),
                'obv': float(talib.OBV(closes, volumes)[-1])
            }
            self.cached_ohlcv = features
            self.last_fetch_time = time.time()
            return features
        except Exception as e:
            print(f"📉 Price data error: {e}")
            return self.cached_ohlcv or {'momentum':0,'volatility':0,'rsi':50,'obv':0}

    async def get_sentiment(self) -> float:
        return 0.0

    def _extract_hft_features(self, order_book: Dict) -> Dict:
        bids, asks = order_book['bids'], order_book['asks']
        if not bids or not asks:
            return self._get_default_hft_features()
        # Use indexing to avoid unpack errors
        bid_prices   = np.array([entry[0] for entry in bids[:5]], dtype=float)
        ask_prices   = np.array([entry[0] for entry in asks[:5]], dtype=float)
        bid_volumes  = np.array([entry[1] for entry in bids[:5]], dtype=float)
        ask_volumes  = np.array([entry[1] for entry in asks[:5]], dtype=float)
        return {
            'bid_ask_spread': ask_prices[0] - bid_prices[0],
            'order_imbalance': bid_volumes.sum() / (ask_volumes.sum() + 1e-6),
            'mid_price_velocity': (ask_prices[0] + bid_prices[0]) / 2 - (self.last_price or 0),
            'cancel_rate': 0.1,
            'fill_rate': 0.85,
            'large_trade_ratio': np.mean(bid_volumes > 10),
            'order_size_entropy': float(-np.sum((bid_volumes/ bid_volumes.sum()) * np.log(bid_volumes/ bid_volumes.sum() + 1e-6))),
            'price_clustering': np.mean((bid_prices % 1.0) < 0.05),
            'volume_ratio': bid_volumes.sum() / (ask_volumes.sum() + 1e-6),
            'lifetime_entropy': 0.9
        }

    def _get_default_hft_features(self) -> Dict:
        return {
            'bid_ask_spread': 0.01,
            'order_imbalance': 1.0,
            'mid_price_velocity': 0.0,
            'cancel_rate': 0.1,
            'fill_rate': 0.85,
            'large_trade_ratio': 0.3,
            'order_size_entropy': 1.2,
            'price_clustering': 0.05,
            'volume_ratio': 1.1,
            'lifetime_entropy': 0.9
        }

    async def shutdown(self):
        print("🛑 Shutting down...")
        self._shutdown_flag = True
        try: await self.market.stop()
        except: pass
        try: await self.exchange.close()
        except: pass
        print("✅ Shutdown complete")

async def main():
    trader = QuantumTrader(symbol='ETH/CAD')
    try:
        await trader.start()
    except KeyboardInterrupt:
        print("\n🛑 Received keyboard interrupt")
    finally:
        await trader.shutdown()

if __name__ == "__main__":
    asyncio.run(main())


