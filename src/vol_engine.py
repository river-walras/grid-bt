import numpy as np
from numba import njit


def second_boundaries(ts_ms: np.ndarray, interval_sec: int = 1):
    if interval_sec < 1:
        raise ValueError("interval_sec must be positive and greater than 1")

    sec = ts_ms // 1000
    bucket = (sec // interval_sec) * interval_sec
    change = np.flatnonzero(bucket[1:] != bucket[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(ts_ms)]
    secs = bucket[starts]
    return secs, starts, ends


def rolling_logret_volatility(
    close_px: np.ndarray,
    *,
    window: int = 300,
) -> np.ndarray:
    """
    Rolling volatility estimate from per-bucket close prices.

    Returns an array `vol` with the same length as `close_px`, where:
    - `vol[t]` is the rolling std-dev of log returns over the last `window` buckets
      (inclusive of bucket t).
    - `vol[0] == 0`.
    """
    close_px = np.asarray(close_px, dtype=np.float64)
    if close_px.ndim != 1:
        raise ValueError("close_px must be a 1D array")
    if close_px.size == 0:
        raise ValueError("close_px cannot be empty")
    if window < 1:
        raise ValueError("window must be >= 1")

    # r[t] = log(close[t] / close[t-1]), with r[0]=0
    r = np.zeros_like(close_px, dtype=np.float64)
    r[1:] = np.log(close_px[1:] / close_px[:-1])

    csum = np.cumsum(np.r_[0.0, r])
    csum2 = np.cumsum(np.r_[0.0, r * r])

    out = np.empty_like(close_px, dtype=np.float64)
    for i in range(close_px.size):
        lo = i - window + 1
        if lo < 0:
            lo = 0
        k = i - lo + 1
        sum_r = csum[i + 1] - csum[lo]
        sum_r2 = csum2[i + 1] - csum2[lo]
        mean_r = sum_r / k
        var = (sum_r2 / k) - (mean_r * mean_r)
        if var < 0.0:
            var = 0.0
        out[i] = np.sqrt(var)
    out[0] = 0.0
    return out


@njit(cache=True)
def _grid_backtest_core_dynamic(
    px,
    qty,
    side,
    secs,
    starts,
    ends,
    tick_sz,
    N,
    grid_pct_arr,
    order_qty,
    fee,
    fill_ratio,
    pos_limit,
    record,
    out_sec,
    out_eq,
    out_inv,
    out_px,
    out_volume,
):
    inv = 0.0
    cash = 0.0
    total_volume = 0.0
    last_price = float(px[0])

    buy_px = np.empty(N, dtype=np.float64)
    sell_px = np.empty(N, dtype=np.float64)
    buy_rem = np.empty(N, dtype=np.float64)
    sell_rem = np.empty(N, dtype=np.float64)

    for t_i in range(len(secs)):
        sec = secs[t_i]
        lo = starts[t_i]
        hi = ends[t_i]

        grid_pct = float(grid_pct_arr[t_i])
        if grid_pct <= 0.0:
            grid_pct = 1e-12

        ref = last_price
        for i in range(N):
            buy_px[i] = (ref * (1.0 - grid_pct) ** (i + 1)) // tick_sz * tick_sz
            sell_px[i] = (ref * (1.0 + grid_pct) ** (i + 1)) // tick_sz * tick_sz

        for i in range(N - 1):
            max_idx = i
            max_val = buy_px[i]
            for j in range(i + 1, N):
                if buy_px[j] > max_val:
                    max_val = buy_px[j]
                    max_idx = j
            if max_idx != i:
                tmp = buy_px[i]
                buy_px[i] = buy_px[max_idx]
                buy_px[max_idx] = tmp

        for i in range(N - 1):
            min_idx = i
            min_val = sell_px[i]
            for j in range(i + 1, N):
                if sell_px[j] < min_val:
                    min_val = sell_px[j]
                    min_idx = j
            if min_idx != i:
                tmp = sell_px[i]
                sell_px[i] = sell_px[min_idx]
                sell_px[min_idx] = tmp

        bq = order_qty if inv < pos_limit else 0.0
        sq = order_qty if inv > -pos_limit else 0.0
        for i in range(N):
            buy_rem[i] = bq
            sell_rem[i] = sq

        for j in range(lo, hi):
            p = float(px[j])
            q = float(qty[j]) * float(fill_ratio)
            if q <= 0.0:
                last_price = p
                continue

            s = side[j]
            if s == -1:
                qleft = q
                for i in range(N):
                    rem = buy_rem[i]
                    if rem <= 0.0:
                        continue
                    lvl = buy_px[i]
                    if p > lvl:
                        break
                    fill = rem if rem < qleft else qleft
                    if fill <= 0.0:
                        break
                    cash -= fill * lvl * (1.0 + fee)
                    inv += fill
                    total_volume += abs(fill)
                    buy_rem[i] = rem - fill
                    qleft -= fill
                    if qleft <= 0.0:
                        break
            elif s == 1:
                qleft = q
                for i in range(N):
                    rem = sell_rem[i]
                    if rem <= 0.0:
                        continue
                    lvl = sell_px[i]
                    if p < lvl:
                        break
                    fill = rem if rem < qleft else qleft
                    if fill <= 0.0:
                        break
                    cash += fill * lvl * (1.0 - fee)
                    inv -= fill
                    total_volume += abs(fill)
                    sell_rem[i] = rem - fill
                    qleft -= fill
                    if qleft <= 0.0:
                        break

            last_price = p

        if record:
            out_sec[t_i] = sec
            out_inv[t_i] = inv
            out_eq[t_i] = cash + inv * last_price
            out_px[t_i] = last_price
            out_volume[t_i] = total_volume

    return cash, inv, last_price, total_volume


def grid_backtest_volatility_numpy(
    ts: np.ndarray,
    px: np.ndarray,
    qty: np.ndarray,
    side: np.ndarray,
    *,
    tick_sz: float = 0.1,
    N: int = 10,
    order_qty: float = 1.0,
    fee: float = 0.0002,
    fill_ratio: float = 1.0,
    pos_limit: float | None = None,
    record: bool = True,
    interval_sec: int = 1,
    vol_window: int = 300,
    vol_mult: float = 1.0,
    min_grid_pct: float = 0.0001,
    max_grid_pct: float = 0.01,
    grid_pct_series: np.ndarray | None = None,
):
    """
    Volatility-adaptive grid backtest.

    Compared to `src.engine.grid_backtest_numpy`, this version replaces the
    fixed `grid_pct` with a per-bucket `grid_pct[t]` derived from rolling
    volatility of per-bucket close-to-close log returns:

        grid_pct[t] = clip(vol_mult * vol[t], min_grid_pct, max_grid_pct)

    If `grid_pct_series` is provided, it is used directly (after clipping) and
    volatility parameters are ignored.
    """
    ts = np.asarray(ts, dtype=np.int64)
    px = np.asarray(px, dtype=np.float64)
    qty = np.asarray(qty, dtype=np.float64)
    side = np.asarray(side, dtype=np.int8)

    if pos_limit is None:
        pos_limit = N * order_qty

    secs, starts, ends = second_boundaries(ts, interval_sec=interval_sec)
    if secs.size == 0:
        raise ValueError("no time buckets produced (empty input?)")

    if grid_pct_series is None:
        close_px = px[ends - 1]
        vol = rolling_logret_volatility(close_px, window=vol_window)
        grid_pct_arr = np.clip(vol_mult * vol, min_grid_pct, max_grid_pct).astype(
            np.float64
        )
    else:
        grid_pct_arr = np.asarray(grid_pct_series, dtype=np.float64)
        if grid_pct_arr.shape != secs.shape:
            raise ValueError(
                "grid_pct_series must have the same length as the bucketed series "
                f"({grid_pct_arr.shape} vs {secs.shape})"
            )
        grid_pct_arr = np.clip(grid_pct_arr, min_grid_pct, max_grid_pct)

    if record:
        out_sec = np.empty(len(secs), dtype=np.int64)
        out_eq = np.empty(len(secs), dtype=np.float64)
        out_inv = np.empty(len(secs), dtype=np.float64)
        out_px = np.empty(len(secs), dtype=np.float64)
        out_volume = np.empty(len(secs), dtype=np.float64)
    else:
        out_sec = np.empty(1, dtype=np.int64)
        out_eq = np.empty(1, dtype=np.float64)
        out_inv = np.empty(1, dtype=np.float64)
        out_px = np.empty(1, dtype=np.float64)
        out_volume = np.empty(1, dtype=np.float64)

    cash, inv, last_price, total_volume = _grid_backtest_core_dynamic(
        px,
        qty,
        side,
        secs,
        starts,
        ends,
        float(tick_sz),
        int(N),
        grid_pct_arr,
        float(order_qty),
        float(fee),
        float(fill_ratio),
        float(pos_limit),
        record,
        out_sec,
        out_eq,
        out_inv,
        out_px,
        out_volume,
    )

    if record:
        return (
            out_sec,
            out_eq,
            out_inv,
            out_px,
            out_volume,
            cash,
            inv,
            last_price,
            total_volume,
            grid_pct_arr,
        )
    return cash, inv, last_price, total_volume, grid_pct_arr
