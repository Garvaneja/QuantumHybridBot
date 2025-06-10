import asyncio
import json
import numpy as np
import websockets
from collections import deque
from typing import Dict, List

class QuantumFeed:
    def __init__(self, exchange: str = 'kraken', symbol: str = 'ETH/USD', depth: int = 25):
        self.symbol = symbol
        self.depth = depth
        self.ws_urls = {
            'kraken': "wss://ws.kraken.com",
        }
        self.ws_url = self.ws_urls.get(exchange)
        if not self.ws_url:
            raise ValueError(f"Unsupported exchange: {exchange}")

        # local order book cache
        self._book = {'bids': deque(maxlen=depth), 'asks': deque(maxlen=depth)}
        self._price = 0.0
        self._connected = False
        self._running = False
        self._lock = asyncio.Lock()

    async def get_order_book(self) -> Dict[str, List[List[float]]]:
        """Thread-safe order book access"""
        async with self._lock:
            return {
                'bids': list(self._book['bids']),
                'asks': list(self._book['asks'])
            }

    def get_optimal_price(self) -> float:
        """Entropy-weighted fair price"""
        if not self._book['bids'] or not self._book['asks']:
            return self._price

        bid_volumes = np.array([v for _, v in self._book['bids']])
        ask_volumes = np.array([v for _, v in self._book['asks']])
        bid_entropy = -np.sum(bid_volumes * np.log(bid_volumes + 1e-10))
        ask_entropy = -np.sum(ask_volumes * np.log(ask_volumes + 1e-10))
        if bid_entropy + ask_entropy == 0:
            return (self._book['bids'][0][0] + self._book['asks'][0][0]) / 2

        bid_price = self._book['bids'][0][0]
        ask_price = self._book['asks'][0][0]
        return (bid_entropy * bid_price + ask_entropy * ask_price) / (bid_entropy + ask_entropy)

    async def _quantum_analyze(self, bids: List[List[float]], asks: List[List[float]]) -> float:
        """Placeholder quantum analysis (correlation)"""
        bid_prices = np.array([p for p, _ in bids])
        ask_prices = np.array([p for p, _ in asks])
        return float(np.corrcoef(bid_prices[:10], ask_prices[:10])[0, 1])

    async def _ws_handler(self):
        """Main WebSocket loop with error handling"""
        self._running = True
        while self._running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self._connected = True
                    print(f"🌐 Connected to WS {self.ws_url} for {self.symbol}")
                    await self._subscribe(ws)
                    while self._running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=10)
                            await self._process_message(msg)
                        except asyncio.TimeoutError:
                            print("⌛ Timeout – waiting for data...")
            except Exception as e:
                print(f"⚠️ Connection error: {e}")
                self._connected = False
                await asyncio.sleep(3)

    async def _subscribe(self, ws):
        """Subscribe to Kraken book channel"""
        sub_msg = {
            "event": "subscribe",
            "pair": [self.symbol],
            "subscription": {"name": "book", "depth": self.depth}
        }
        print(f"🌐 Subscribing to Kraken book for {self.symbol}")
        await ws.send(json.dumps(sub_msg))

    async def _process_message(self, msg: str):
        """Parse Kraken WS messages and merge snapshots + updates"""
        try:
            data = json.loads(msg)
            # Debug raw message
            print("🔍 RAW WS MSG:", data)
            # skip subscription/heartbeat
            if isinstance(data, dict) and data.get("event"):
                return
            # expect list messages with book data
            if isinstance(data, list) and len(data) >= 2:
                payload = data[1]
                bids = payload.get("bs", []) + payload.get("b", [])
                asks = payload.get("as", []) + payload.get("a", [])
                await self._update_book(bids, asks)
        except Exception as e:
            print(f"⚠️ _process_message error: {e}")

    async def _update_book(self, bids: List[List[str]], asks: List[List[str]]):
        """Update internal book and run quantum analysis"""
        async with self._lock:
            try:
                # only price & size
                processed_bids = [[float(x[0]), float(x[1])] for x in bids]
                processed_asks = [[float(x[0]), float(x[1])] for x in asks]
                self._book['bids'] = deque(sorted(processed_bids, reverse=True), maxlen=self.depth)
                self._book['asks'] = deque(sorted(processed_asks), maxlen=self.depth)
                if self._book['bids'] and self._book['asks']:
                    q_score = await self._quantum_analyze(self._book['bids'], self._book['asks'])
                    print(f"⚛️ Quantum Score: {q_score:.2f}")
            except Exception as e:
                print(f"⚠️ Book update failed: {e}")

    async def start(self):
        """Start the feed loop"""
        asyncio.create_task(self._ws_handler())

    def is_connected(self) -> bool:
        """Return True if WS is connected"""
        return getattr(self, '_connected', False)

    async def stop(self):
        """Graceful shutdown"""
        self._running = False
        while self._connected:
            await asyncio.sleep(0.1)

