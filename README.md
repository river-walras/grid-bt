HFT Grid Backtest Task
=============================

练习目标
--------
- 运行 `python src/engine.py` 回测 OKX Swap 逐笔数据。
- 核心指标：**最大化 Final Equity**、同时**提高 Avg Daily Turnover**。

关键文件
--------
- `src/engine.py`：含数据加载、Numba 加速的逐秒撮合网格策略。
- `main.py`：主函数
