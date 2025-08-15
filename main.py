# -*- coding: utf-8 -*-
"""
Универсальная визуализация задачи ARC из JSON-файла.
Читает файл D:\\tom\\ARC.txt, содержащий строку вида {"train": [...], "test": [...]}
"""
import matplotlib
matplotlib.use('TkAgg')  # Безопасный бэкенд


import json
from typing import List, Dict, Any, Optional
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

# ---- Палитра ARC (0..9) ----
ARC_COLORS = [
    "#FFFFFF",  # 0 background
    "#3B82F6",  # 1 blue
    "#EF4444",  # 2 red
    "#10B981",  # 3 emerald
    "#F59E0B",  # 4 amber
    "#9CA3AF",  # 5 gray
    "#8B5CF6",  # 6 violet
    "#F97316",  # 7 orange
    "#14B8A6",  # 8 teal
    "#6B7280",  # 9 slate
]
cmap = ListedColormap(ARC_COLORS)
bounds = np.arange(-0.5, len(ARC_COLORS) + 0.5, 1.0)
norm = BoundaryNorm(bounds, cmap.N)


def simple_predict(train: List[Dict[str, Any]], test_input: List[List[int]]) -> List[List[int]]:
    """
    Простейший предиктор для демонстрационных целей.
    Возвращает вход как предсказание.
    """
    # Здесь могла бы быть реализация алгоритма ARC.
    # Пока возвращаем сам вход без изменений.
    return test_input

def draw_grid(ax: plt.Axes, array: List[List[int]], title: Optional[str] = None) -> None:
    """Рисует дискретную сетку с границами клеток."""
    grid = np.array(array, dtype=int)
    ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest")

    h, w = grid.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", linewidth=0.8, color="#DDDDDD")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    if title:
        ax.set_title(title, fontsize=11, pad=6)

def visualize_arc_task(data: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """Визуализирует все пары train (input/output) и test."""
    train = data.get("train", [])
    tests = data.get("test", [])

    n_train = len(train)
    n_test = len(tests)
    total_rows = n_train + n_test
    total_cols = 2  # input / output

    fig_h = max(2.0 * total_rows, 3.0)
    fig_w = 8.0
    fig, axes = plt.subplots(
        nrows=total_rows, ncols=total_cols, figsize=(fig_w, fig_h),
        squeeze=False, layout="constrained"
    )

    row = 0
    for i, pair in enumerate(train, start=1):
        ax_in, ax_out = axes[row][0], axes[row][1]
        draw_grid(ax_in, pair["input"], title=f"train #{i} — input")
        draw_grid(ax_out, pair["output"], title=f"train #{i} — output")
        row += 1

    for j, pair in enumerate(tests, start=1):
        ax_in, ax_out = axes[row][0], axes[row][1]
        draw_grid(ax_in, pair["input"], title=f"test #{j} — input")

        prediction = simple_predict(train, pair["input"])
        print(f"\nTest #{j} prediction:\n{prediction}")

        if "output" in pair and pair["output"] is not None:
            print(f"Test #{j} correct output:\n{pair['output']}")
            draw_grid(ax_out, pair["output"], title=f"test #{j} — output")
        else:
            ax_out.axis("off")
            ax_out.set_title(f"test #{j} — output (нет ответа)")
        row += 1

    fig.suptitle("ARC: визуализация входов и выходов", fontsize=13)
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

if __name__ == "__main__":
    file_path = os.path.join(os.path.dirname(__file__), "ARC.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        task_data = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Ошибка чтения JSON: {e}")

    visualize_arc_task(task_data)
