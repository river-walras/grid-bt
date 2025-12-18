import numpy as np
import os
import duckdb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, UTC

def load_data(
    symbol: str,
    conn: duckdb.DuckDBPyConnection,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    order_by_ts: bool = True,
):
    """
    Load data from Parquet files for the given symbol.

    Parameters:
    symbol (str): The trading symbol to load data for.

    Returns:
    duckdb.DuckDBPyRelation: A DuckDB relation containing the loaded data.
    """
    if os.path.exists(f"{symbol}_trades.parquet"):
        path = f"{symbol}_trades.parquet"
        sql = f"SELECT * FROM read_parquet('{path}')"
    else:

        path = f"/share/okx_data/swap/trades/daily/{symbol}/*.parquet"
        sql = f"SELECT CASE WHEN side = 'buy' THEN 1 WHEN side = 'sell' THEN -1 END AS side, price, size, created_time FROM read_parquet('{path}')"
        if start_date:
            sql += f" WHERE created_time >= '{int(start_date.timestamp() * 1000)}'"
        if end_date:
            sql += f" AND created_time < '{int(end_date.timestamp() * 1000)}'"
        if order_by_ts:
            sql += " ORDER BY created_time ASC"

    data = conn.execute(sql).fetchnumpy()
    total_days = (end_date - start_date).days if start_date and end_date else None
    return data, total_days

def resample_minute(
    ts_sec: np.ndarray, eq: np.ndarray, inv: np.ndarray, px: np.ndarray
):
    """
    Down-sample second-level series to per-minute snapshots by taking the last
    observation inside each minute bucket.
    """
    ts_sec = np.asarray(ts_sec, dtype=np.int64)
    eq = np.asarray(eq, dtype=np.float64)
    inv = np.asarray(inv, dtype=np.float64)
    px = np.asarray(px, dtype=np.float64)

    if ts_sec.size == 0:
        raise ValueError("empty time series cannot be resampled")

    minutes = ts_sec // 60
    change = np.flatnonzero(minutes[1:] != minutes[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(minutes)]

    out_ts = np.empty(len(starts), dtype=np.int64)
    out_eq = np.empty(len(starts), dtype=np.float64)
    out_inv = np.empty(len(starts), dtype=np.float64)
    out_px = np.empty(len(starts), dtype=np.float64)

    for i, hi in enumerate(ends):
        idx = hi - 1  # use last observation inside the bucket
        out_ts[i] = minutes[idx] * 60
        out_eq[i] = eq[idx]
        out_inv[i] = inv[idx]
        out_px[i] = px[idx]

    return out_ts, out_eq, out_inv, out_px


def second_boundaries(ts_ms: np.ndarray, interval_sec: int = 1):
    if interval_sec < 1:
        raise ValueError("interval_sec must be positive and greater than 1")

    sec = ts_ms // 1000
    bucket = (sec // interval_sec) * interval_sec
    change = (
        np.flatnonzero(bucket[1:] != bucket[:-1]) + 1
    )  # indices where interval changes
    starts = np.r_[0, change]  # include the first index
    ends = np.r_[change, len(ts_ms)]
    secs = bucket[starts]
    return secs, starts, ends


def plot_minute_equity(
    ts_sec, eq, inv, px, *, title: str | None = None, save_path: str | None = None
):
    """
    Plot minute-level price/equity on the first axes and inventory on the second.
    """
    m_ts, m_eq, m_inv, m_px = resample_minute(ts_sec, eq, inv, px)
    times = mdates.date2num([datetime.fromtimestamp(t, UTC) for t in m_ts])

    fig, (ax_eq, ax_inv) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))

    ax_price_top = ax_eq.twinx()
    ln_eq = ax_eq.plot(times, m_eq, label="Equity", color="tab:blue", linewidth=1.5)
    ln_price_top = ax_price_top.plot(
        times, m_px, label="Price", color="0.5", linewidth=1.0, alpha=0.7
    )
    ax_eq.set_ylabel("Equity")
    ax_price_top.set_ylabel("Price")
    ax_eq.grid(True, linestyle="--", alpha=0.3)
    ax_eq.legend(
        ln_eq + ln_price_top,
        [l.get_label() for l in ln_eq + ln_price_top],
        loc="upper left",
    )

    ax_price_bottom = ax_inv.twinx()
    ln_inv = ax_inv.plot(
        times, m_inv, label="Position (Qty)", color="tab:blue", linewidth=1.2
    )
    ln_price_bottom = ax_price_bottom.plot(
        times, m_px, label="Price", color="0.5", linewidth=1.0, alpha=0.5
    )
    ax_inv.set_ylabel("Position (Qty)")
    ax_price_bottom.set_ylabel("Price")
    ax_inv.grid(True, linestyle="--", alpha=0.3)
    ax_inv.legend(
        ln_inv + ln_price_bottom,
        [l.get_label() for l in ln_inv + ln_price_bottom],
        loc="upper left",
    )
    ax_inv.set_xlabel("Time (UTC)")

    formatter = mdates.DateFormatter("%m-%d %H:%M")
    ax_inv.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
    else:
        fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    else:
        plt.show()

    return fig
