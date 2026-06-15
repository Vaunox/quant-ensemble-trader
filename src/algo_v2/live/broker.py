import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseBroker(ABC):
    """
    Abstract Base Class for all brokerage APIs (Zerodha, IBKR, Upstox, etc.).
    Any broker you plug into the AI must implement these exact methods.
    """

    @abstractmethod
    def get_cash_balance(self) -> float:
        """Returns the total available cash (INR) currently sitting in the account."""
        pass

    @abstractmethod
    def get_holdings(self) -> dict:
        """Returns a dictionary of current holdings. Format: {'TICKER': quantity}"""
        pass

    @abstractmethod
    def get_latest_prices(self, tickers: list) -> dict:
        """Returns the exact current execution price for the requested tickers."""
        pass

    @abstractmethod
    def place_market_order(self, ticker: str, shares: int, current_price: float) -> bool:
        """
        Executes a market order.
        Positive shares = BUY, Negative shares = SELL.
        Returns True if successful, False if rejected (e.g. insufficient funds).
        """
        pass


class DummyBroker(BaseBroker):
    """
    A local JSON-backed broker that simulates the exact mechanics of a live broker.
    It tracks its own state independently of the RL algorithm to ensure the Sim-to-Real gap is bridged.

    Trades are written to an append-only trades_YYYY-MM.jsonl log (one JSON object
    per line) rather than growing the state file — prevents the state file from
    bloating to thousands of entries over months of daily runs.
    """

    def __init__(self, initial_capital=1000000.0, state_file="dummy_broker_state.json", fee_pct=0.0002):
        self.state_file = state_file
        self.fee_pct = fee_pct

        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                self.state = json.load(f)
            # Migrate: drop legacy transaction_history from state if present
            self.state.pop("transaction_history", None)
        else:
            self.state = {"cash": initial_capital, "holdings": {}}
            self._save_state()

    def _save_state(self):
        # Atomic write: write to a temp file then rename so a crash never
        # leaves a half-written (and therefore corrupt) state file.
        dir_ = os.path.dirname(os.path.abspath(self.state_file))
        with tempfile.NamedTemporaryFile("w", dir=dir_, delete=False, suffix=".tmp") as tmp:
            json.dump(self.state, tmp, indent=4)
            tmp_path = tmp.name
        os.replace(tmp_path, self.state_file)

    def _log_trade(self, record: dict):
        """Append one trade record to the monthly JSONL trade log."""
        month_tag = datetime.now().strftime("%Y-%m")
        log_dir = os.path.dirname(os.path.abspath(self.state_file))
        log_path = os.path.join(log_dir, f"trades_{month_tag}.jsonl")
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def get_cash_balance(self) -> float:
        return self.state["cash"]

    def get_holdings(self) -> dict:
        return self.state["holdings"]

    def get_latest_prices(self, tickers: list) -> dict:
        # In the DummyBroker, the executor script will fetch the prices via yfinance
        # and pass them into the place_order function instead.
        # A real broker would query its own live websocket here.
        raise NotImplementedError("DummyBroker relies on the Executor to provide prices.")

    def place_market_order(self, ticker: str, shares: int, current_price: float) -> bool:
        if shares == 0:
            return True

        cost_value = abs(shares) * current_price
        fee = cost_value * self.fee_pct
        total_transaction_cost = cost_value + fee

        # BUY Logic
        if shares > 0:
            if self.state["cash"] < total_transaction_cost:
                logger.warning("BROKER REJECTED: Insufficient funds to buy %d of %s", shares, ticker)
                return False

            self.state["cash"] -= total_transaction_cost
            current_held = self.state["holdings"].get(ticker, 0)
            self.state["holdings"][ticker] = current_held + shares

            self._log_trade(
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "action": "BUY",
                    "shares": shares,
                    "price": current_price,
                    "fee": round(fee, 4),
                }
            )
            self._save_state()
            return True

        # SELL Logic
        if shares < 0:
            sell_qty = abs(shares)
            current_held = self.state["holdings"].get(ticker, 0)

            if current_held < sell_qty:
                logger.warning("BROKER REJECTED: Cannot sell %d of %s (holding %d)", sell_qty, ticker, current_held)
                return False

            net_proceeds = cost_value - fee
            self.state["cash"] += net_proceeds

            self.state["holdings"][ticker] -= sell_qty
            if self.state["holdings"][ticker] == 0:
                del self.state["holdings"][ticker]

            self._log_trade(
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "action": "SELL",
                    "shares": sell_qty,
                    "price": current_price,
                    "fee": round(fee, 4),
                }
            )
            self._save_state()
            return True
