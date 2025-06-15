import asyncio
import curses
import time
from functools import partial
from typing import Optional
import contextlib

from main import QuantumTrader


class Dashboard:
    """Simple curses-based UI to monitor QuantumTrader."""

    def __init__(self, symbol: str = "ETH/USD"):
        self.trader = QuantumTrader(symbol=symbol)
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def _start_trader(self):
        try:
            await self.trader.start()
        except asyncio.CancelledError:
            pass
        finally:
            await self.trader.shutdown()

    def _ui_loop(self, stdscr: curses.window):
        curses.curs_set(0)
        stdscr.nodelay(True)
        while not self.trader._shutdown_flag:
            stdscr.erase()
            stdscr.addstr(0, 0, "QuantumTrader Dashboard (press q to quit)")
            stdscr.addstr(2, 0, f"Symbol: {self.trader.symbol}")
            stdscr.addstr(3, 0, f"Last Price : {self.trader.last_price}")
            stdscr.addstr(4, 0, f"Last Signal: {self.trader.last_signal}")
            stdscr.addstr(5, 0, f"Errors     : {self.trader._consecutive_errors}/{self.trader.max_errors}")
            stdscr.refresh()
            ch = stdscr.getch()
            if ch == ord("q"):
                asyncio.run_coroutine_threadsafe(self.trader.shutdown(), self.loop)
                break
            time.sleep(1)

    async def run(self):
        self.loop = asyncio.get_running_loop()
        trader_task = asyncio.create_task(self._start_trader())
        await asyncio.to_thread(curses.wrapper, partial(self._ui_loop))
        trader_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await trader_task


if __name__ == "__main__":
    asyncio.run(Dashboard().run())
