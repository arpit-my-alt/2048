# 2048 AI Solver

An AI agent that plays the game **2048** on its own, using the **Expectimax** search algorithm combined with a hand-tuned set of heuristics. The project includes:

- `2048Solver.ipynb` — the original notebook: game engine, heuristics, Expectimax search, and a matplotlib-based run of the solver (heuristic value / search-call count vs. move number).
- `pygame_2048_visualizer.py` — a standalone script that runs the exact same AI on an animated, interactive 2048 board instead of static plots.

## How it works

### Game engine
The board is a 4×4 NumPy array. Each move (`UP`, `DOWN`, `LEFT`, `RIGHT`) is simulated by sliding every row/column toward the target edge and merging equal adjacent tiles (`slide_and_merge`). A move is only legal if it changes the board.

### Heuristics
Since 2048 has no simple win condition to search toward directly, each board state is scored by a weighted combination of six properties:

| Heuristic | What it measures |
|---|---|
| **Empty cells** | More free space = more room to maneuver |
| **Monotonicity** | Rows/columns trending in one direction, which is easier to merge |
| **Smoothness** | How close adjacent tile values are (in log2 space) — smaller gaps merge more easily |
| **Corner** | Rewards keeping the highest tile pinned in a corner |
| **Gradient** | Rewards a "snake"-like weighted layout that funnels tiles toward one corner |
| **Potential merges** | Number of adjacent tile pairs that can merge right now |

These are combined into a single score:

```python
score = 12*empty + 15*mono + 5*smooth + 150*corner + 20*merges + 10*gradient
```

### Expectimax search
2048's randomness (new tiles spawning as 2 with 90% probability or 4 with 10%) makes it a natural fit for **Expectimax** rather than plain Minimax:

- **Maximizing nodes** — the AI picks the move (`UP`/`DOWN`/`LEFT`/`RIGHT`) that leads to the best expected outcome.
- **Chance nodes** — every empty cell is expanded with both possible tile spawns, weighted by their spawn probability, and averaged.

The search runs to a depth of 3, automatically increasing to depth 4 once a `1024` tile appears on the board, trading speed for foresight in the late game.

## Getting started

### Requirements
```bash
pip install numpy matplotlib pygame
```

### Run the notebook
Open `2048Solver.ipynb` in Jupyter and run all cells. The final cell plays a full game and plots the heuristic value and number of Expectimax calls after every move.

```bash
jupyter notebook 2048Solver.ipynb
```

### Run the visual simulation
```bash
python pygame_2048_visualizer.py
```

This opens a live, color-coded 2048 board (classic tile palette) and watches the AI play automatically, with a sidebar showing score, move count, best tile, and current AI speed.

**Controls:**

| Key | Action |
|---|---|
| `SPACE` | Pause / resume the AI |
| `R` | Restart with a new game |
| `↑` | Speed up (shorter delay between moves) |
| `↓` | Slow down (longer delay between moves) |
| `ESC` / `Q` | Quit |

## Project structure
```
.
├── 2048Solver.ipynb           # Notebook: engine, heuristics, Expectimax, matplotlib run
├── pygame_2048_visualizer.py  # Standalone animated pygame version of the same AI
└── README.md
```

## Notes & possible improvements
- The Expectimax branching factor grows quickly with the number of empty cells, so raw Python recursion gets slow in the early game; caching/transposition tables, alpha-beta-style pruning, or bitboard representations (as used in high-performance 2048 solvers) would speed this up considerably.
- Heuristic weights were hand-tuned; they could be optimized automatically (e.g. via a genetic algorithm or hill climbing over many simulated games).
- The pygame version currently redraws instantly on each move; adding tile slide/merge animations would make it feel closer to the real game.
