"""Вариант: чётный номер зачётной книжки -> функция sin(x).

Запуск:
    python lab1.py                  # архитектура берётся из results/best_arch.json

    python lab1.py --hidden 8 8     # своя архитектура
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

FUNC = "sin"   # чётный номер зачётки
SEED = 42
N_SAMPLES = 100_000
X_MIN, X_MAX = -2 * np.pi, 2 * np.pi


# Данные
def make_data(func: str = FUNC, n: int = N_SAMPLES, seed: int = SEED):
    rng = np.random.default_rng(seed)
    x = rng.uniform(X_MIN, X_MAX, size=(n, 1)).astype(np.float32)
    y = (np.sin(x) if func == "sin" else np.cos(x)).astype(np.float32)
    return x, y


# Разделение на обучающий / проверочный
def split_data(x, y, train_ratio: float = 0.8, seed: int = SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    n_train = int(len(x) * train_ratio)
    tr, va = idx[:n_train], idx[n_train:]
    return x[tr], y[tr], x[va], y[va]


# Модель
def build_model(hidden, seed=SEED):
    torch.manual_seed(seed)
    layers, in_f = [], 1
    for h in hidden:
        layers += [nn.Linear(in_f, h), nn.Tanh()]
        in_f = h
    layers.append(nn.Linear(in_f, 1))  # выход без активации (регрессия)
    return nn.Sequential(*layers)


# Обучение и проверка
def train_model(model, x_tr, y_tr, x_va, y_va, epochs=40, batch_size=256,
                lr=1e-2, verbose=True):
    torch.manual_seed(SEED)
    loss_fn = nn.MSELoss()                                   # регрессия -> MSE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam, шаг lr
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va)),
        batch_size=1024, shuffle=False)

    history = []
    for epoch in range(1, epochs + 1):
        # цикл обучения
        model.train()
        total, count = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
            count += len(xb)
        train_loss = total / count

        # цикл проверки
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                total += loss_fn(model(xb), yb).item() * len(xb)
                count += len(xb)
        val_loss = total / count

        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if verbose:
            print(f"Эпоха {epoch:3d}/{epochs} | train MSE = {train_loss:.6f} | val MSE = {val_loss:.6f}")
    return history


def plot_loss(epochs, values, title, path, color):
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, values, color=color, marker="o", markersize=3)
    plt.yscale("log")
    plt.xlabel("Эпоха")
    plt.ylabel("Ошибка (MSE)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, nargs="+", default=None,
                    help="размеры скрытых слоёв, напр. 16 16")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.hidden is None:
        best = os.path.join(args.out, "best_arch.json")
        if os.path.exists(best):
            with open(best, encoding="utf-8") as fh:
                args.hidden = json.load(fh)["hidden"]
            print(f"Архитектура из {best}: {args.hidden}")
        else:
            args.hidden = [16, 16]
            print("best_arch.json не найден, используется [16, 16]")

    # данные
    x, y = make_data(FUNC)
    x_tr, y_tr, x_va, y_va = split_data(x, y)
    print(f"Обучающий набор: {len(x_tr)}, проверочный: {len(x_va)}")

    # график функции
    xs = np.linspace(X_MIN, X_MAX, 1000)
    ys = np.sin(xs) if FUNC == "sin" else np.cos(xs)
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, label=f"{FUNC}(x)")
    plt.scatter(x_tr[:300], y_tr[:300], s=6, c="tab:orange", label="обучающие точки (300 из 80000)")
    plt.scatter(x_va[:100], y_va[:100], s=6, c="tab:green", label="проверочные точки (100 из 20000)")
    plt.xlabel("x"); plt.ylabel("y"); plt.grid(True, alpha=0.3); plt.legend()
    plt.title(f"Функция {FUNC}(x) и выборки")
    plt.tight_layout()
    plt.savefig(f"{args.out}/function.png", dpi=150)
    plt.close()

    # модель
    model = build_model(args.hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Параметров: {n_params}")

    # обучение + проверка
    history = train_model(model, x_tr, y_tr, x_va, y_va,
                          epochs=args.epochs, batch_size=args.batch, lr=args.lr)

    # история в csv
    df = pd.DataFrame(history)
    df[["epoch", "train_loss"]].to_csv(f"{args.out}/train_history.csv", index=False)
    df[["epoch", "val_loss"]].to_csv(f"{args.out}/val_history.csv", index=False)
    df.to_csv(f"{args.out}/history.csv", index=False)

    # графики Ошибка - Эпоха
    plot_loss(df["epoch"], df["train_loss"], "Обучение: ошибка по эпохам",
              f"{args.out}/train_loss.png", "tab:blue")
    plot_loss(df["epoch"], df["val_loss"], "Проверка: ошибка по эпохам",
              f"{args.out}/val_loss.png", "tab:red")

    # предсказание сети против истинной функции
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(xs.astype(np.float32)).unsqueeze(1)).numpy().ravel()
    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, label="истинная функция")
    plt.plot(xs, pred, "--", label="предсказание сети")
    plt.xlabel("x"); plt.ylabel("y"); plt.grid(True, alpha=0.3); plt.legend()
    plt.title("Результат аппроксимации")
    plt.tight_layout()
    plt.savefig(f"{args.out}/prediction.png", dpi=150)
    plt.close()

    torch.save(model.state_dict(), f"{args.out}/model.pt")
    print(f"\nИтог: train MSE = {history[-1]['train_loss']:.6f}, "
          f"val MSE = {history[-1]['val_loss']:.6f}")
    print(f"Файлы сохранены в папку '{args.out}/'")


if __name__ == "__main__":
    main()
