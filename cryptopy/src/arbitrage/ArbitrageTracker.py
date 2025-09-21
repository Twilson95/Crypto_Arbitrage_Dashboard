import time
import json
from pathlib import Path
from statistics import pstdev


class ArbitrageTracker:
    """
    Tracks arbitrage opportunities across refresh cycles.

    Each opportunity is identified by a key built from:
      (arb_type, currency_tuple, exchanges_tuple, path_tuple_or_None)

    On every call to `update()`:
        * Active opportunities are updated or created.
        * Disappeared / non‑profitable ones are finalised.
        * A snapshot ("event": "update") of all *current* active
          opportunities is appended to disk so progress is never lost.
    """

    def __init__(self, outfile="arbitrage_lifetimes.jsonl"):
        self.outfile = Path(outfile)
        self.active = {}  # key -> record dict

    # ------------------------------------------------------------------ #
    def update(self, arbitrage_list, now=None):
        """
        :param arbitrage_list: iterable of arbitrage_data dicts.
        :param now: optional timestamp override (float).
        """
        now = now if now is not None else time.time()
        seen = set()

        # --- process current opportunities ---
        if not arbitrage_list:
            return

        for arb in arbitrage_list:
            profit = float(arb["summary_header"]["total_profit"])
            if profit <= 0:
                continue  # treat non‑profitable as dead

            key = self._build_key(arb)
            seen.add(key)

            if key not in self.active:
                # new opportunity
                self.active[key] = {
                    "start_ts": now,
                    "last_ts": now,
                    "ticks": 1,
                    "profits": [profit],
                    "last_profit": profit,
                    "max_profit": profit,
                    "min_profit": profit,
                    "cum_time_profit": 0.0,
                    "meta": arb["summary_header"],
                }
            else:
                rec = self.active[key]
                dt = now - rec["last_ts"]
                if dt > 0:
                    rec["cum_time_profit"] += rec["last_profit"] * dt
                rec["last_ts"] = now
                rec["ticks"] += 1
                rec["profits"].append(profit)
                rec["last_profit"] = profit
                rec["max_profit"] = max(rec["max_profit"], profit)
                rec["min_profit"] = min(rec["min_profit"], profit)

        # --- finalise vanished opportunities ---
        dead_keys = [k for k in list(self.active) if k not in seen]
        for k in dead_keys:
            self._finalise(k, reason="disappeared", end_ts=now)

        # --- snapshot all still‑active opportunities to disk ---
        self._snapshot_active(now)

    def close_all(self):
        """Force‑finalise all active opportunities (e.g. on shutdown)."""
        now = time.time()
        for k in list(self.active):
            self._finalise(k, reason="shutdown", end_ts=now)

    # ------------------------------------------------------------------ #
    def _build_key(self, arb):
        sh = arb["summary_header"]
        currency = tuple(sh.get("currency", []))
        exchanges = tuple(sh.get("exchanges_used", []))
        path = tuple(tuple(p) for p in arb.get("path", [])) or None
        arb_type = "cycle" if path else "simple"
        return (arb_type, currency, exchanges, path)

    def _snapshot_active(self, now):
        """Write one JSON line per active opportunity (heartbeat)."""
        for key, rec in self.active.items():
            lifetime = rec["last_ts"] - rec["start_ts"]
            avg_profit = sum(rec["profits"]) / rec["ticks"]
            stdev = pstdev(rec["profits"]) if rec["ticks"] > 1 else 0.0
            snap = {
                "event": "update",
                "key": key,
                "timestamp": now,
                "start_ts": rec["start_ts"],
                "last_ts": rec["last_ts"],
                "lifetime_seconds": lifetime,
                "observations": rec["ticks"],
                "last_profit": rec["last_profit"],
                "avg_profit": avg_profit,
                "max_profit": rec["max_profit"],
                "min_profit": rec["min_profit"],
                "profit_stdev": stdev,
                "meta": rec["meta"],
            }
            self._append_json(snap)

    def _finalise(self, key, reason, end_ts):
        rec = self.active.pop(key)
        # add final segment to time‑weighted accumulator
        dt_final = end_ts - rec["last_ts"]
        if dt_final > 0:
            rec["cum_time_profit"] += rec["last_profit"] * dt_final

        lifetime = rec["last_ts"] - rec["start_ts"]
        avg_profit = sum(rec["profits"]) / rec["ticks"]
        time_weighted_avg = (
            rec["cum_time_profit"] / lifetime if lifetime > 0 else rec["last_profit"]
        )
        stdev = pstdev(rec["profits"]) if rec["ticks"] > 1 else 0.0
        sharpe_like = (avg_profit / stdev) if stdev else None

        out_record = {
            "event": "finalise",
            "key": key,
            "reason": reason,
            "start_ts": rec["start_ts"],
            "end_ts": rec["last_ts"],
            "lifetime_seconds": lifetime,
            "observations": rec["ticks"],
            "avg_profit": avg_profit,
            "time_weighted_avg_profit": time_weighted_avg,
            "max_profit": rec["max_profit"],
            "min_profit": rec["min_profit"],
            "last_profit": rec["last_profit"],
            "profit_stdev": stdev,
            "sharpe_like": sharpe_like,
            "meta": rec["meta"],
        }
        self._append_json(out_record)

    def _append_json(self, record):
        with self.outfile.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
