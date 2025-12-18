import duckdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, UTC

# load data from /share/okx_data/swap/trades/daily/{symbol}/*.parquet


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
    path = f"/share/okx_data/swap/trades/daily/{symbol}/*.parquet"
    sql = f"SELECT CASE WHEN side = 'buy' THEN 1 WHEN side = 'sell' THEN -1 END AS side, price, size, created_time FROM read_parquet('{path}')"
    if start_date:
        sql += f" WHERE created_time >= '{int(start_date.timestamp() * 1000)}'"
    if end_date:
        sql += f" AND created_time < '{int(end_date.timestamp() * 1000)}'"
    if order_by_ts:
        sql += " ORDER BY created_time ASC"
    data = conn.execute(sql).fetchnumpy()
    return data


def second_boundaries(ts_ms: np.ndarray):
    sec = ts_ms // 1000
    change = np.flatnonzero(sec[1:] != sec[:-1]) + 1  # indices where second changes
    starts = np.r_[0, change]  # include the first index
    ends = np.r_[change, len(ts_ms)]
    secs = sec[starts]
    return secs, starts, ends


def grid_backtest_by_second_numpy(
    ts: np.ndarray,
    px: np.ndarray,
    qty: np.ndarray,
    side: np.ndarray,
    *,
    tick_sz: float = 0.1,
    N: int = 10,
    grid_pct: float = 0.001,
    order_qty: float = 1.0,  # 每档数量（币）
    fee: float = 0.0002,
    fill_ratio: float = 1.0,
    pos_limit: float | None = None,
    record: bool = True,
):
    ts = np.asarray(ts, dtype=np.int64)
    px = np.asarray(px, dtype=np.float64)
    qty = np.asarray(qty, dtype=np.float64)
    side = np.asarray(side, dtype=np.int8)

    if pos_limit is None:
        pos_limit = N * order_qty

    secs, starts, ends = second_boundaries(ts)

    inv = 0.0
    cash = 0.0
    last_price = float(px[0])

    # 复用数组，避免每秒分配
    buy_px = np.empty(N, dtype=np.float64)
    buy_rem = np.empty(N, dtype=np.float64)
    sell_px = np.empty(N, dtype=np.float64)
    sell_rem = np.empty(N, dtype=np.float64)

    def rebuild(ref, inv):
        for i in range(N):
            buy_px[i] = (ref * (1 - grid_pct) ** (i + 1)) // tick_sz * tick_sz
            sell_px[i] = (ref * (1 + grid_pct) ** (i + 1)) // tick_sz * tick_sz

        # 排序：buy 从高到低，sell 从低到高
        buy_px.sort()
        buy_px[:] = buy_px[::-1]
        sell_px.sort()

        # 每秒重置剩余量（秒内不补）
        bq = order_qty if inv < pos_limit else 0.0
        sq = order_qty if inv > -pos_limit else 0.0
        buy_rem.fill(bq)
        sell_rem.fill(sq)

    # 初始网格：用第一笔价格做 ref（不前视）
    rebuild(last_price, inv)

    if record:
        out_sec = np.empty(len(secs), dtype=np.int64)
        out_eq = np.empty(len(secs), dtype=np.float64)
        out_inv = np.empty(len(secs), dtype=np.float64)
        out_px = np.empty(len(secs), dtype=np.float64)

    for t_i, (sec, lo, hi) in enumerate(zip(secs, starts, ends)):
        # 秒切换：用上一秒末价 last_price 作为 ref（不前视）
        rebuild(last_price, inv)

        # 撮合本秒 trades
        for j in range(lo, hi):
            p = float(px[j])
            q = float(qty[j]) * float(fill_ratio)
            if q <= 0:
                last_price = p
                continue

            s = side[j]

            if s == -1:
                # 主动卖：打 buy levels（要求 trade price <= level）
                qleft = q
                for i in range(N):
                    rem = buy_rem[i]
                    if rem <= 0:
                        continue
                    lvl = buy_px[i]
                    if p > lvl:
                        break
                    fill = rem if rem < qleft else qleft
                    if fill <= 0:
                        break
                    cash -= fill * lvl * (1.0 + fee)
                    inv += fill
                    buy_rem[i] = rem - fill
                    qleft -= fill
                    if qleft <= 0:
                        break

            elif s == 1:
                # 主动买：打 sell levels（要求 trade price >= level）
                qleft = q
                for i in range(N):
                    rem = sell_rem[i]
                    if rem <= 0:
                        continue
                    lvl = sell_px[i]
                    if p < lvl:
                        break
                    fill = rem if rem < qleft else qleft
                    if fill <= 0:
                        break
                    cash += fill * lvl * (1.0 - fee)
                    inv -= fill
                    sell_rem[i] = rem - fill
                    qleft -= fill
                    if qleft <= 0:
                        break

            last_price = p

        if record:
            out_sec[t_i] = sec
            out_inv[t_i] = inv
            out_eq[t_i] = cash + inv * last_price
            out_px[t_i] = last_price

    if record:
        return out_sec, out_eq, out_inv, out_px, cash, inv, last_price
    return cash, inv, last_price


def resample_minute(ts_sec: np.ndarray, eq: np.ndarray, inv: np.ndarray, px: np.ndarray):
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


def plot_minute_equity(ts_sec, eq, inv, px, *, title: str | None = None, save_path: str | None = None):
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
    ax_eq.legend(ln_eq + ln_price_top, [l.get_label() for l in ln_eq + ln_price_top], loc="upper left")

    ax_price_bottom = ax_inv.twinx()
    ln_inv = ax_inv.plot(times, m_inv, label="Position (Qty)", color="tab:blue", linewidth=1.2)
    ln_price_bottom = ax_price_bottom.plot(
        times, m_px, label="Price", color="0.5", linewidth=1.0, alpha=0.5
    )
    ax_inv.set_ylabel("Position (Qty)")
    ax_price_bottom.set_ylabel("Price")
    ax_inv.grid(True, linestyle="--", alpha=0.3)
    ax_inv.legend(ln_inv + ln_price_bottom, [l.get_label() for l in ln_inv + ln_price_bottom], loc="upper left")
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


if __name__ == "__main__":
    conn = duckdb.connect()

    symbol = "BTC-USDT-SWAP"
    data = load_data(
        symbol, conn, start_date=datetime(2025, 10, 1), end_date=datetime(2025, 11, 1)
    )

    ts = data["created_time"]
    px = data["price"]
    qty = data["size"]
    side = data["side"]

    out_sec, out_eq, out_inv, out_px, cash, inv, last_price = grid_backtest_by_second_numpy(
        ts,
        px,
        qty,
        side,
        tick_sz=0.1,
        N=5,
        grid_pct=0.0002,
        order_qty=0.01,
        fee=0,
        fill_ratio=1.0,
        pos_limit=0.05,
        record=True,
    )

    # for s, eq, iv in zip(out_sec, out_eq, out_inv):
    #     print(f"Second: {s}, Equity: {eq:.2f}, Inventory: {iv:.4f}")

    # print(
    #     f"Final Cash: {cash:.2f}, Final Inventory: {inv:.4f}, Last Price: {last_price:.2f}"
    # )

    plot_minute_equity(
        out_sec,
        out_eq,
        out_inv,
        out_px,
        title=f"{symbol} Grid Backtest",
        save_path=f"{symbol}_grid_backtest.png",
    )
