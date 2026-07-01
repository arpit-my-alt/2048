"""
2048 AI Solver — Pygame Visual Simulation
==========================================

Watches the same Expectimax + heuristics AI from `2048Solver.ipynb` play
2048 in real time, rendered as an actual animated game board instead of
matplotlib plots.

Controls
--------
    SPACE   Pause / resume the AI
    R       Restart with a new game
    UP      Speed up (fewer ms between moves)
    DOWN    Slow down (more ms between moves)
    ESC / Q Quit

Run
---
    pip install pygame numpy
    python pygame_2048_visualizer.py
"""

import random
from copy import deepcopy

import numpy as np
import pygame

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
GRID_SIZE = 4

CELL_SIZE = 110
CELL_MARGIN = 12
BOARD_PADDING = 20
BOARD_PIXELS = GRID_SIZE * CELL_SIZE + (GRID_SIZE + 1) * CELL_MARGIN

SIDEBAR_WIDTH = 260
WINDOW_WIDTH = BOARD_PIXELS + 2 * BOARD_PADDING + SIDEBAR_WIDTH
WINDOW_HEIGHT = BOARD_PIXELS + 2 * BOARD_PADDING

DEFAULT_MOVE_DELAY_MS = 120  # time between AI moves

# Colors (classic 2048 palette)
COLOR_BG = (250, 248, 239)
COLOR_BOARD_BG = (187, 173, 160)
COLOR_EMPTY_CELL = (205, 193, 180)
COLOR_TEXT_DARK = (119, 110, 101)
COLOR_TEXT_LIGHT = (249, 246, 242)

TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (60, 58, 50),
    8192: (40, 38, 30),
}

FONT_NAME = "arial"


# --------------------------------------------------------------------------
# Core 2048 game logic (ported from 2048Solver.ipynb)
# --------------------------------------------------------------------------
def slide_and_merge(line):
    line = line[line != 0]
    merged = []
    skip = False
    for i in range(len(line)):
        if skip:
            skip = False
            continue
        if i + 1 < len(line) and line[i] == line[i + 1]:
            merged.append(line[i] * 2)
            skip = True
        else:
            merged.append(line[i])
    merged = np.array(merged)
    merged = np.pad(merged, (0, GRID_SIZE - len(merged)), "constant")
    return merged


def simulate_move(board, direction):
    new_board = np.zeros_like(board)
    if direction == LEFT:
        for i in range(GRID_SIZE):
            new_board[i, :] = slide_and_merge(board[i, :])
    elif direction == RIGHT:
        for i in range(GRID_SIZE):
            new_board[i, ::-1] = slide_and_merge(board[i, ::-1])
    elif direction == UP:
        for j in range(GRID_SIZE):
            new_board[:, j] = slide_and_merge(board[:, j])
    elif direction == DOWN:
        for j in range(GRID_SIZE):
            new_board[::-1, j] = slide_and_merge(board[::-1, j])
    return new_board


def count_empty_cells(board):
    return np.count_nonzero(board == 0)


def monotonicity(board):
    mono = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1):
            if board[i, j] >= board[i, j + 1]:
                mono += 1
            if board[j, i] >= board[j + 1, i]:
                mono += 1
    return mono


def smoothness(board):
    smooth = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if board[i, j] != 0:
                val = np.log2(board[i, j])
                for (di, dj) in [(0, 1), (1, 0)]:
                    ni, nj = i + di, j + dj
                    if ni < GRID_SIZE and nj < GRID_SIZE and board[ni, nj] != 0:
                        smooth -= abs(val - np.log2(board[ni, nj]))
    return smooth


def max_tile_in_corner(board):
    max_tile = np.max(board)
    return 1 if (
        board[0, 0] == max_tile
        or board[0, 3] == max_tile
        or board[3, 0] == max_tile
        or board[3, 3] == max_tile
    ) else 0


def gradient_heuristic(board):
    weights = np.array(
        [
            [15, 14, 13, 12],
            [8, 9, 10, 11],
            [7, 6, 5, 4],
            [0, 1, 2, 3],
        ]
    )
    return np.sum(board * weights)


def potential_merges(board):
    merges = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE - 1):
            if board[i, j] == board[i, j + 1] and board[i, j] != 0:
                merges += 1
            if board[j, i] == board[j + 1, i] and board[j, i] != 0:
                merges += 1
    return merges


def heuristic(board):
    empty = count_empty_cells(board)
    mono = monotonicity(board)
    smooth = smoothness(board)
    corner = max_tile_in_corner(board)
    merges = potential_merges(board)
    gradient = gradient_heuristic(board)

    return (
        12 * empty
        + 15 * mono
        + 5 * smooth
        + 150 * corner
        + 20 * merges
        + 10 * gradient
    )


def game_over(board):
    for move in [UP, DOWN, LEFT, RIGHT]:
        if not np.array_equal(board, simulate_move(board, move)):
            return False
    return True


def get_search_depth(board):
    return 4 if np.max(board) >= 1024 else 3


def expectimax(board, depth, is_maximizing):
    if depth == 0 or game_over(board):
        return heuristic(board)

    if is_maximizing:
        max_eval = -np.inf
        for move in [UP, DOWN, LEFT, RIGHT]:
            new_board = simulate_move(board, move)
            if not np.array_equal(board, new_board):
                max_eval = max(max_eval, expectimax(new_board, depth - 1, False))
        return max_eval
    else:
        empty_positions = list(zip(*np.where(board == 0)))
        if not empty_positions:
            return 0
        total_eval = 0
        for pos in empty_positions:
            for tile, prob in ((2, 0.9), (4, 0.1)):
                new_board = deepcopy(board)
                new_board[pos] = tile
                total_eval += prob * expectimax(new_board, depth - 1, True)
        return total_eval / len(empty_positions)


def get_best_move(board):
    best_move, best_eval = None, -np.inf
    depth = get_search_depth(board)
    for move in [UP, DOWN, LEFT, RIGHT]:
        new_board = simulate_move(board, move)
        if not np.array_equal(board, new_board):
            eval_ = expectimax(new_board, depth, False)
            if eval_ > best_eval:
                best_eval, best_move = eval_, move
    return best_move if best_move is not None else random.choice([UP, DOWN, LEFT, RIGHT])


def add_random_tile(board):
    empty_positions = list(zip(*np.where(board == 0)))
    if empty_positions:
        pos = random.choice(empty_positions)
        board[pos] = 2 if random.random() < 0.9 else 4


def new_game():
    board = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    add_random_tile(board)
    add_random_tile(board)
    return board


# --------------------------------------------------------------------------
# Pygame rendering
# --------------------------------------------------------------------------
def draw_board(screen, board, fonts, score, moves, best_tile, paused, delay_ms):
    screen.fill(COLOR_BG)

    board_rect = pygame.Rect(BOARD_PADDING, BOARD_PADDING, BOARD_PIXELS, BOARD_PIXELS)
    pygame.draw.rect(screen, COLOR_BOARD_BG, board_rect, border_radius=8)

    font_big, font_med, font_small = fonts

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            value = int(board[i, j])
            x = BOARD_PADDING + CELL_MARGIN + j * (CELL_SIZE + CELL_MARGIN)
            y = BOARD_PADDING + CELL_MARGIN + i * (CELL_SIZE + CELL_MARGIN)
            color = TILE_COLORS.get(value, TILE_COLORS[8192])
            pygame.draw.rect(screen, color, (x, y, CELL_SIZE, CELL_SIZE), border_radius=6)

            if value != 0:
                text_color = COLOR_TEXT_DARK if value <= 4 else COLOR_TEXT_LIGHT
                font = font_big if value < 100 else (font_med if value < 1000 else font_small)
                text_surf = font.render(str(value), True, text_color)
                text_rect = text_surf.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                screen.blit(text_surf, text_rect)

    # Sidebar
    sidebar_x = BOARD_PIXELS + 2 * BOARD_PADDING + 10
    title = font_med.render("2048 AI Solver", True, COLOR_TEXT_DARK)
    screen.blit(title, (sidebar_x, 30))

    lines = [
        f"Score: {score}",
        f"Moves: {moves}",
        f"Best tile: {best_tile}",
        f"Speed: {delay_ms} ms/move",
        "",
        "SPACE  pause/resume",
        "R      restart",
        "UP/DOWN speed",
        "ESC/Q  quit",
    ]
    if paused:
        lines.insert(0, "-- PAUSED --")

    for idx, line in enumerate(lines):
        surf = font_small.render(line, True, COLOR_TEXT_DARK)
        screen.blit(surf, (sidebar_x, 90 + idx * 30))


def main():
    pygame.init()
    pygame.display.set_caption("2048 — Expectimax AI Solver")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    font_big = pygame.font.SysFont(FONT_NAME, 44, bold=True)
    font_med = pygame.font.SysFont(FONT_NAME, 28, bold=True)
    font_small = pygame.font.SysFont(FONT_NAME, 20)
    fonts = (font_big, font_med, font_small)

    board = new_game()
    moves = 0
    paused = False
    delay_ms = DEFAULT_MOVE_DELAY_MS
    finished = False

    MOVE_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(MOVE_EVENT, delay_ms)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    board = new_game()
                    moves = 0
                    finished = False
                    paused = False
                elif event.key == pygame.K_UP:
                    delay_ms = max(10, delay_ms - 20)
                    pygame.time.set_timer(MOVE_EVENT, delay_ms)
                elif event.key == pygame.K_DOWN:
                    delay_ms = min(1000, delay_ms + 20)
                    pygame.time.set_timer(MOVE_EVENT, delay_ms)
            elif event.type == MOVE_EVENT and not paused and not finished:
                if game_over(board):
                    finished = True
                else:
                    move = get_best_move(board)
                    board = simulate_move(board, move)
                    add_random_tile(board)
                    moves += 1
                    if np.max(board) >= 2048:
                        finished = True

        score = int(np.sum(board))
        best_tile = int(np.max(board))
        draw_board(screen, board, fonts, score, moves, best_tile, paused or finished, delay_ms)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()
