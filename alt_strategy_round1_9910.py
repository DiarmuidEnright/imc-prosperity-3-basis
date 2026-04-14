"""
alt_strategy_round1.py — Alternative Round 1 strategies.

Drop-in replacement for solution_round1.py. Submit directly to the IMC
platform or backtester to A/B test against the main Round 1 strategy.

─────────────────────────────────────────────────────────────────────────────
PHILOSOPHY — what's different and why
─────────────────────────────────────────────────────────────────────────────

AltAshCoatedOsmiumTrader  (vs AshCoatedOsmiumTrader)
  The main strategy assumes bot walls sit at a fixed distance from FV.
  This version detects them dynamically from the live order book on every
  tick, then posts 1 tick inside the dominant level. Two additions:
    1. Multi-level quoting: 65% of capacity at the primary (inner) level,
       35% at a secondary (outer) level. More fills, less missed volume.
    2. Insider amplification: when a known insider bot (Olivia / Caesar /
       Vladimir / Camilla) is trading, take at FV regardless of inventory
       direction. Used with success by Alpha Animals and Frankfurt Hedgehogs
       in Prosperity 3.
    3. Three-tier inventory skew (none / medium / heavy) instead of the
       continuous formula — easier to tune.

AltIntarianPepperRootTrader  (vs IntarianPepperRootTrader)
  INTARIAN_PEPPER_ROOT has a perfectly linear trend (+1/1000 timestamps).
  Spike mean-reversion or EMA crossover would work against the trend, so
  the core strategy remains buy-and-hold at max position. One addition:
    - Insider boost: when a named insider is detected net-buying, we post
      a more aggressive bid (1 tick below best ask instead of 2) to fill
      our remaining capacity faster. No selling under any circumstance.
"""

from __future__ import annotations
import json
import math
from collections import deque
from datamodel import Order, OrderDepth, TradingState

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEBUG = False  # NEVER True in competition submissions

ASH_LIMIT       = 80
INTARIAN_LIMIT  = 80

INSIDER_NAMES: frozenset[str] = frozenset({"Olivia", "Caesar", "Vladimir", "Camilla"})

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _best_bid(depth: OrderDepth) -> int | None:
    return max(depth.buy_orders) if depth.buy_orders else None

def _best_ask(depth: OrderDepth) -> int | None:
    return min(depth.sell_orders) if depth.sell_orders else None

def _dominant_bid(depth: OrderDepth, min_vol: int = 5) -> int | None:
    """Highest bid price among levels with volume >= min_vol (bot wall proxy)."""
    if not depth.buy_orders:
        return None
    candidates = [p for p, v in depth.buy_orders.items() if v >= min_vol]
    return max(candidates) if candidates else max(depth.buy_orders)

def _dominant_ask(depth: OrderDepth, min_vol: int = 5) -> int | None:
    """Lowest ask price among levels with abs(volume) >= min_vol (bot wall proxy)."""
    if not depth.sell_orders:
        return None
    candidates = [p for p, v in depth.sell_orders.items() if abs(v) >= min_vol]
    return min(candidates) if candidates else min(depth.sell_orders)

def _insider_direction(state: TradingState, symbol: str) -> int:
    """+1 = insiders net-bought, -1 = net-sold, 0 = no signal."""
    trades = state.market_trades.get(symbol, [])
    bought = sum(t.quantity for t in trades if t.buyer in INSIDER_NAMES)
    sold   = sum(t.quantity for t in trades if t.seller in INSIDER_NAMES)
    if bought > sold:
        return 1
    if sold > bought:
        return -1
    return 0

def _no_wash(orders: list[Order]) -> list[Order]:
    """Remove any (buy, sell) pairs where buy.price >= sell.price."""
    buys  = [(i, o) for i, o in enumerate(orders) if o.quantity > 0]
    sells = [(i, o) for i, o in enumerate(orders) if o.quantity < 0]
    bad: set[int] = set()
    for bi, bo in buys:
        for si, so in sells:
            if bo.price >= so.price:
                bad.add(bi)
                bad.add(si)
    return [o for i, o in enumerate(orders) if i not in bad]


# ─────────────────────────────────────────────────────────────────────────────
# AltAshCoatedOsmiumTrader
# ─────────────────────────────────────────────────────────────────────────────

class AltAshCoatedOsmiumTrader:
    """
    ASH_COATED_OSMIUM alternative: dynamic bot-wall detection + two-level
    quoting + insider amplification + three-tier inventory skew.

    Data context (confirmed from submissions up to 116042):
      - FV permanently ~10 000.
      - Bot walls at mid ± 8 (dominant: bid 9992, ask 10008). Median spread 16.
      - Position limit: 80.

    Two-level passive quote design (empirically validated):
      - Primary (65%): just inside bot wall (dom_bid+1 / dom_ask-1).
      - Secondary (35%): at the bot wall itself (dom_bid / dom_ask cap).
    Inner levels (FV±2, FV±4) were tested and made performance WORSE:
    they intercept bot bids that would have filled the outer wall at better
    prices, reducing avg spread from 9.78 to 4.37 ticks.
    """

    FV:           int   = 10_000
    LIMIT:        int   = ASH_LIMIT
    MAX_OFFSET:   int   = 8      # bot wall at FV ± 8; quotes in [FV-8, FV+8]
    PRIMARY_FRAC: float = 0.65

    MED_TIER:  float = 0.30
    HIGH_TIER: float = 0.65

    def run(self, state: TradingState, td: dict) -> list[Order]:
        sym   = "ASH_COATED_OSMIUM"
        depth = state.order_depths.get(sym)
        if not depth:
            return []

        fv       = self.FV
        pos      = state.position.get(sym, 0)
        buy_cap  = self.LIMIT - pos
        sell_cap = self.LIMIT + pos
        insider  = _insider_direction(state, sym)
        orders: list[Order] = []

        # -- 1. Aggressive takes ------------------------------------------
        for ask in sorted(depth.sell_orders):
            if buy_cap <= 0:
                break
            if ask < fv:
                vol = min(abs(depth.sell_orders[ask]), buy_cap)
                orders.append(Order(sym, ask, vol))
                buy_cap -= vol
            elif ask == fv:
                if pos < 0 or insider > 0:
                    vol = min(abs(depth.sell_orders[ask]), buy_cap)
                    orders.append(Order(sym, ask, vol))
                    buy_cap -= vol

        for bid in sorted(depth.buy_orders, reverse=True):
            if sell_cap <= 0:
                break
            if bid > fv:
                vol = min(depth.buy_orders[bid], sell_cap)
                orders.append(Order(sym, bid, -vol))
                sell_cap -= vol
            elif bid == fv:
                if pos > 0 or insider < 0:
                    vol = min(depth.buy_orders[bid], sell_cap)
                    orders.append(Order(sym, bid, -vol))
                    sell_cap -= vol

        # -- 2. Three-tier inventory skew ---------------------------------
        inv_ratio = abs(pos) / self.LIMIT
        if inv_ratio <= self.MED_TIER:
            skew = 0
        elif inv_ratio <= self.HIGH_TIER:
            skew = 1 if pos > 0 else -1
        else:
            skew = 2 if pos > 0 else -2

        # -- 3. Dynamic bot-wall detection --------------------------------
        dom_bid = _dominant_bid(depth)
        dom_ask = _dominant_ask(depth)

        raw_primary_bid = (dom_bid + 1) if (dom_bid is not None and dom_bid < fv) else (fv - 1)
        primary_bid = max(fv - self.MAX_OFFSET, min(fv - 1, raw_primary_bid)) - skew

        raw_primary_ask = (dom_ask - 1) if (dom_ask is not None and dom_ask > fv) else (fv + 1)
        primary_ask = min(fv + self.MAX_OFFSET, max(fv + 1, raw_primary_ask)) - skew

        secondary_bid = max(fv - self.MAX_OFFSET, primary_bid - 2)
        secondary_ask = min(fv + self.MAX_OFFSET, primary_ask + 2)

        if secondary_bid >= primary_bid:
            secondary_bid = max(fv - self.MAX_OFFSET, primary_bid - 1)
        if secondary_ask <= primary_ask:
            secondary_ask = min(fv + self.MAX_OFFSET, primary_ask + 1)

        primary_bid   = max(fv - self.MAX_OFFSET, min(fv - 1, primary_bid))
        primary_ask   = min(fv + self.MAX_OFFSET, max(fv + 1, primary_ask))
        secondary_bid = max(fv - self.MAX_OFFSET, min(fv - 1, secondary_bid))
        secondary_ask = min(fv + self.MAX_OFFSET, max(fv + 1, secondary_ask))

        # -- 4. Split capacity across two levels --------------------------
        if buy_cap > 0:
            prim_qty = max(1, round(buy_cap * self.PRIMARY_FRAC))
            sec_qty  = buy_cap - prim_qty
            orders.append(Order(sym, primary_bid, prim_qty))
            if sec_qty > 0:
                orders.append(Order(sym, secondary_bid, sec_qty))

        if sell_cap > 0:
            prim_qty = max(1, round(sell_cap * self.PRIMARY_FRAC))
            sec_qty  = sell_cap - prim_qty
            orders.append(Order(sym, primary_ask, -prim_qty))
            if sec_qty > 0:
                orders.append(Order(sym, secondary_ask, -sec_qty))

        result = _no_wash(orders)

        if DEBUG:
            print(
                f"[ALT-ASH] pos={pos} skew={skew} insider={insider} "
                f"dom_bid={dom_bid} dom_ask={dom_ask} "
                f"pbid={primary_bid} pask={primary_ask}"
            )
        return result


# ─────────────────────────────────────────────────────────────────────────────
# AltIntarianPepperRootTrader
# ─────────────────────────────────────────────────────────────────────────────

class AltIntarianPepperRootTrader:
    """
    INTARIAN_PEPPER_ROOT alternative: buy-and-hold trend follower with
    insider-boosted fill aggression.

    Data context:
      - Price rises exactly +1/1000 timestamps per day, monotonically.
      - Position limit: 25. Optimal: hold 25 units all day.
      - No mean-reversion signal; spike/EMA approaches work against the trend.

    Enhancement over main strategy:
      - When a named insider is detected net-buying, post bid at best_ask
        (cross the spread to fill immediately). Otherwise post at best_ask - 1.
        Never sell under any circumstance.
    """

    LIMIT: int = INTARIAN_LIMIT

    def run(self, state: TradingState, td: dict) -> list[Order]:
        sym   = "INTARIAN_PEPPER_ROOT"
        depth = state.order_depths.get(sym)
        if not depth:
            return []

        pos     = state.position.get(sym, 0)
        buy_cap = self.LIMIT - pos
        if buy_cap <= 0:
            return []

        insider = _insider_direction(state, sym)
        orders: list[Order] = []

        # -- 1. Take available asks ---------------------------------------
        for ask in sorted(depth.sell_orders):
            if buy_cap <= 0:
                break
            vol = min(abs(depth.sell_orders[ask]), buy_cap)
            orders.append(Order(sym, ask, vol))
            buy_cap -= vol

        # -- 2. Passive bid for remaining capacity -----------------------
        if buy_cap > 0:
            ba = _best_ask(depth)
            bb = _best_bid(depth)
            if ba is not None:
                # Insider buying: cross the spread to fill immediately
                post_bid = ba if insider > 0 else ba - 1
            elif bb is not None:
                post_bid = bb + 1
            else:
                post_bid = None

            if post_bid is not None:
                orders.append(Order(sym, post_bid, buy_cap))

        if DEBUG:
            print(f"[ALT-INTARIAN] pos={pos} buy_cap={buy_cap} insider={insider}")

        return orders


# ─────────────────────────────────────────────────────────────────────────────
# Trader — exchange entry point
# ─────────────────────────────────────────────────────────────────────────────

class Trader:
    """
    Drop-in Trader class using the alternative Round 1 strategies.
    Submit this file directly via the IMC platform or backtester.
    """

    def __init__(self) -> None:
        self._ash      = AltAshCoatedOsmiumTrader()
        self._intarian = AltIntarianPepperRootTrader()

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        td: dict = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except Exception:
                pass

        orders: dict[str, list[Order]] = {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            result = self._ash.run(state, td)
            if result:
                orders["ASH_COATED_OSMIUM"] = result

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            result = self._intarian.run(state, td)
            if result:
                orders["INTARIAN_PEPPER_ROOT"] = result

        trader_data = json.dumps(td, separators=(",", ":"))
        return orders, 0, trader_data
