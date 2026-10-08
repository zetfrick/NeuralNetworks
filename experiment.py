"""
Функция sin(x).
Запуск: python experiment.py
Перебирает число пар (Linear + Tanh) и ширину слоя, печатает таблицу
и сохраняет её в results/experiment.csv
"""
import argparse
import json
import os

import pandas as pd

from lab1 import build_model, make_data, split_data, train_model

CONFIGS = [
    [2], [4], [8], [16], [32],           # ширина при 1 слое
    [2, 2], [4, 4], [8, 8], [16, 16],    # 2 слоя
    [4, 4, 4], [8, 8, 8], [16, 16, 16],  # 3 слоя
]
THRESHOLD = 1e-3  # считаем задачу решённой, если val MSE < 1e-3

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=40)
args = ap.parse_args()

x, y = make_data()
x_tr, y_tr, x_va, y_va = split_data(x, y)

rows = []
for hidden in CONFIGS:
    model = build_model(hidden)
    n_params = sum(p.numel() for p in model.parameters())
    h = train_model(model, x_tr, y_tr, x_va, y_va, epochs=args.epochs, verbose=False)
    rows.append({"hidden": str(hidden), "pairs": len(hidden), "params": n_params,
                 "val_mse": h[-1]["val_loss"], "ok": h[-1]["val_loss"] < THRESHOLD})
    print(f"{str(hidden):14s} параметров={n_params:5d}  val MSE={h[-1]['val_loss']:.6f}")

df = pd.DataFrame(rows).sort_values("params")
os.makedirs("results", exist_ok=True)
df.to_csv("results/experiment.csv", index=False)
print("\n", df.to_string(index=False))
good = df[df["ok"]]
if len(good):
    best = good.iloc[0]
    print("\nМинимальная подходящая архитектура:", best["hidden"], "параметров:", best["params"])
    with open("results/best_arch.json", "w", encoding="utf-8") as fh:
        json.dump({"hidden": json.loads(best["hidden"]), "params": int(best["params"])}, fh)
    print("Сохранено в results/best_arch.json - lab1.py возьмёт её автоматически.")
else:
    print("\nНи одна архитектура не достигла порога — увеличьте эпохи или размер сети.")
