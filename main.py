from datetime import datetime
import duckdb
from src import (
    load_data,
    grid_backtest_numpy,
    plot_minute_equity,
    grid_backtest_volatility_numpy,
)


def volatility_bench():
    conn = duckdb.connect()

    symbol = "BTC-USDT-SWAP"
    data, total_days = load_data(
        symbol, conn, start_date=datetime(2025, 1, 1), end_date=datetime(2025, 5, 1)
    )

    ts = data["created_time"]
    px = data["price"]
    qty = data["size"]
    side = data["side"]

    tick_sz = 0.1
    N = 5
    min_grid_pct = 0.0002
    max_grid_pct = 0.002
    order_qty = 0.01
    fee = 0.0
    fill_ratio = 1.0
    pos_limit = 0.05
    interval_sec = 1

    (
        out_sec,
        out_eq,
        out_inv,
        out_px,
        _,
        _,
        _,
        _,
        total_volume,
        grid_pct_arr,
    ) = grid_backtest_volatility_numpy(
        ts,
        px,
        qty,
        side,
        tick_sz=tick_sz,
        N=N,
        order_qty=order_qty,
        fee=fee,
        fill_ratio=fill_ratio,
        pos_limit=pos_limit,
        min_grid_pct=min_grid_pct,
        max_grid_pct=max_grid_pct,
        record=True,
        vol_window=300,
        vol_mult=1.0,
        interval_sec=interval_sec,
    )

    print(
        f"Final Equity: {out_eq[-1]:.2f}, Avg Daily Turnover: {total_volume / total_days / pos_limit:.2f}, total_days: {total_days}"
    )

    plot_minute_equity(
        out_sec,
        out_eq,
        out_inv,
        out_px,
        title=f"{symbol} Grid Volatility Backtest",
        save_path=f"{symbol}_grid_volatility_backtest.png",
    )


def basic_bench():
    conn = duckdb.connect()

    symbol = "BTC-USDT-SWAP"
    data, total_days = load_data(
        symbol, conn, start_date=datetime(2025, 1, 1), end_date=datetime(2025, 5, 1)
    )

    ts = data["created_time"]
    px = data["price"]
    qty = data["size"]
    side = data["side"]

    tick_sz = 0.1
    N = 5
    grid_pct = 0.0002
    order_qty = 0.01
    fee = 0.0
    fill_ratio = 1.0
    pos_limit = 0.05
    interval_sec = 1

    (
        out_sec,
        out_eq,
        out_inv,
        out_px,
        _,
        _,
        _,
        _,
        total_volume,
    ) = grid_backtest_numpy(
        ts,
        px,
        qty,
        side,
        tick_sz=tick_sz,
        N=N,
        grid_pct=grid_pct,
        order_qty=order_qty,
        fee=fee,
        fill_ratio=fill_ratio,
        pos_limit=pos_limit,
        record=True,
        interval_sec=interval_sec,
    )

    print(
        f"Final Equity: {out_eq[-1]:.2f}, Avg Daily Turnover: {total_volume / total_days / pos_limit:.2f}, total_days: {total_days}"
    )

    plot_minute_equity(
        out_sec,
        out_eq,
        out_inv,
        out_px,
        title=f"{symbol} Grid Backtest",
        save_path=f"{symbol}_grid_backtest.png",
    )

def main():
    print("Running Basic Grid Backtest...")
    basic_bench()
    print("\nRunning Volatility Grid Backtest...")
    volatility_bench()

if __name__ == "__main__":
    main()
