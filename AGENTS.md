# AGENTS.md

This repo's agent instructions live in **[CLAUDE.md](./CLAUDE.md)** — read that
file first. It covers the project purpose, scope, layout, build/run commands,
and the conventions that keep the autograd risk path intact.

Quick orientation:

- **Goal:** price option-free Treasury & Agency bonds using a tensor curve
  framework (libtorch = C++ PyTorch); risk via `torch::autograd`.
- **Scope now:** non-callable fixed-coupon only. Muni/corporate/callable later.
- **Build:** `cmake -DCMAKE_PREFIX_PATH=/abs/path/to/libtorch -B build -S . && cmake --build build -j`
- **Data:** `python3 python/fetch_treasury.py` (treasury.gov, no API key).
- **Test:** `cd build && ctest --output-on-failure`

Design details and roadmap: **[docs/tensor_curve_framework.md](./docs/tensor_curve_framework.md)**.
