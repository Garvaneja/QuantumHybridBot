# Architecture

The project is organized into modular components:

- **core** – trading logic and analytics (hybrid models, risk management).
- **execution** – routers that choose brokers and order types.
- **main.py** – orchestration entry point used to run the bot.

Data flows from the market feed through feature extraction and signal generation
before passing risk checks and being executed by the router.
