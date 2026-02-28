
# ============================================================
#  ОПТИМИЗАЦИЯ ВЕСОВ ОБЛИГАЦИОННОГО ПОРТФЕЛЯ
#  scipy SLSQP — аналог "Поиска решения" Excel
#
#  Входные файлы (генерируются bond_screener.py и ofz_screener.py):
#    ofz_shortlist.csv       — 6 ОФЗ
#    corporate_shortlist.csv — 10 корпоративных облигаций
#
#  Задача:
#    max  Σ w_i * YTM_i
#    s.t. Σ w_i * D_i  = H         (иммунизация дюрации)
#         Σ w_i        = 1         (полное инвестирование)
#         Σ w_i[ОФЗ]  ≥ 35%       (мин. доля госбумаг)
#         w_i          ≤ 15%       (макс. вес одной бумаги)
#         w_i[корп]    ≤ 10%       (макс. вес корпоратива)
#         w_i          ≥ 0         (нет коротких позиций)
# ============================================================

import pandas as pd
import numpy as np
from scipy.optimize import minimize

# ------------------------------------------------------------
# ПАРАМЕТРЫ — меняй здесь
# ------------------------------------------------------------
H           = 3.0   # целевая дюрация (горизонт), лет
H_TOL       = 0.05  # допуск: дюрация попадёт в [H-tol, H+tol]
W_MAX       = 0.15  # макс. вес одной бумаги
W_MAX_CORP  = 0.10  # макс. вес одного корпоратива
W_MIN_OFZ   = 0.35  # мин. суммарный вес ОФЗ
# ------------------------------------------------------------

# ============================================================
# БЛОК 0: ЗАГРУЗКА И ОБЪЕДИНЕНИЕ ДАННЫХ
# ============================================================

ofz  = pd.read_csv("ofz_shortlist.csv",       encoding="utf-8-sig")
corp = pd.read_csv("corporate_shortlist.csv",  encoding="utf-8-sig")

ofz["тип"]  = "ОФЗ"
corp["тип"] = "Корпоратив"

ofz  = ofz.rename(columns={"Индикативная доходность, %": "ytm", "dur_years": "dur"})
corp = corp.rename(columns={"Индикативная доходность, %": "ytm", "dur_years": "dur"})

keep = ["Бумага", "ISIN", "ytm", "dur", "Ликвидность", "тип"]
df = pd.concat([ofz[keep], corp[keep]], ignore_index=True)

n       = len(df)
ytm_pct = df["ytm"].values          # в процентах (14.5, 20.0 ...)
ytm     = ytm_pct / 100             # в долях — для оптимизатора
dur     = df["dur"].values
is_ofz  = (df["тип"] == "ОФЗ").values
is_corp = ~is_ofz

print(f"Загружено бумаг: {n}  ({is_ofz.sum()} ОФЗ + {is_corp.sum()} корп.)")
print(f"YTM диапазон   : {ytm_pct.min():.2f}% — {ytm_pct.max():.2f}%")
print(f"Дюрация диапаз.: {dur.min():.2f} — {dur.max():.2f} лет\n")

# ============================================================
# БЛОК 1: ПОСТАНОВКА ЗАДАЧИ ОПТИМИЗАЦИИ
# ============================================================

def objective(w):
    """Минимизируем отрицательный YTM → максимизируем YTM."""
    return -np.dot(w, ytm)

constraints = [
    # 1. Сумма весов = 1
    {"type": "eq",
     "fun":  lambda w: np.sum(w) - 1.0,
     "jac":  lambda w: np.ones(n)},

    # 2a. Нижняя граница иммунизации: D_port >= H - tol
    {"type": "ineq",
     "fun":  lambda w: np.dot(w, dur) - (H - H_TOL),
     "jac":  lambda w: dur},

    # 2b. Верхняя граница иммунизации: D_port <= H + tol
    {"type": "ineq",
     "fun":  lambda w: (H + H_TOL) - np.dot(w, dur),
     "jac":  lambda w: -dur},

    # 3. Минимальная доля ОФЗ
    {"type": "ineq",
     "fun":  lambda w: np.dot(w, is_ofz.astype(float)) - W_MIN_OFZ,
     "jac":  lambda w: is_ofz.astype(float)},
]

# Индивидуальные границы весов
lb = np.zeros(n)
ub = np.where(is_corp, W_MAX_CORP, W_MAX)
bounds = list(zip(lb, ub))

# ============================================================
# БЛОК 2: ЗАПУСК ОПТИМИЗАЦИИ
# Несколько стартовых точек — берём лучшее решение
# ============================================================

best_result = None
starts = [
    np.ones(n) / n,                           # равные веса
    np.where(is_ofz, W_MIN_OFZ / is_ofz.sum(),
             (1 - W_MIN_OFZ) / is_corp.sum()), # равно внутри групп
    np.random.dirichlet(np.ones(n)),           # случайный старт
]

for w0 in starts:
    # Проецируем начальное приближение в допустимую область
    w0 = np.clip(w0, lb, ub)
    w0 = w0 / w0.sum()

    res = minimize(
        fun=objective,
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 2000, "disp": False},
    )
    if res.success and (best_result is None or res.fun < best_result.fun):
        best_result = res

if best_result is None or not best_result.success:
    raise RuntimeError(f"Solver не нашёл решения: {best_result.message}")

w_opt = best_result.x
w_opt = np.where(w_opt < 1e-4, 0.0, w_opt)   # обнуляем < 0.01%
w_opt = w_opt / w_opt.sum()                   # нормируем до 1

print("✅ Оптимальное решение найдено!")

# ============================================================
# БЛОК 3: ВЫВОД РЕЗУЛЬТАТОВ
# ============================================================

df["вес_%"]      = w_opt * 100
df["вклад_ytm"]  = w_opt * ytm_pct
df["вклад_dur"]  = w_opt * dur

port_ytm = df["вклад_ytm"].sum()   # уже в %
port_dur = df["вклад_dur"].sum()
w_ofz_s  = df.loc[is_ofz,  "вес_%"].sum() / 100
w_corp_s = df.loc[is_corp, "вес_%"].sum() / 100
hhi      = np.sum(w_opt**2)

print()
print("=" * 80)
print(f"  ОПТИМАЛЬНЫЙ ПОРТФЕЛЬ  |  H = {H} лет  |  max YTM  (SLSQP)")
print("=" * 80)
print(f"  {'#':<3} {'Бумага':<42} {'Тип':<12} {'Вес':>7}  {'YTM':>6}  {'Дюр':>6}")
print("-" * 80)

active = df[df["вес_%"] > 0.01].sort_values("тип")
for i, (_, r) in enumerate(active.iterrows(), 1):
    print(f"  {i:<3} {str(r['Бумага'])[:41]:<42} {r['тип']:<12} "
          f"{r['вес_%']:>6.1f}%  "
          f"{r['ytm']:>5.2f}%  "
          f"{r['dur']:>5.2f} л.")

print("=" * 80)
print(f"\n  ХАРАКТЕРИСТИКИ ПОРТФЕЛЯ:")
print(f"  {'YTM портфеля:':<35} {port_ytm:.2f}%")
print(f"  {'Дюрация:':<35} {port_dur:.3f} лет  "
      f"(цель {H} ± {H_TOL})")
print(f"  {'Доля ОФЗ:':<35} {w_ofz_s*100:.1f}%  "
      f"(мин {W_MIN_OFZ*100:.0f}%)")
print(f"  {'Доля корпоративных:':<35} {w_corp_s*100:.1f}%")
print(f"  {'Активных позиций:':<35} {(df['вес_%'] > 0.01).sum()} из {n}")
print(f"  {'HHI (концентрация):':<35} {hhi:.4f}  "
      f"({'низкая' if hhi < 0.10 else 'умеренная' if hhi < 0.18 else 'высокая'})")

print(f"\n  ПРОВЕРКА ОГРАНИЧЕНИЙ:")
checks = [
    ("Σ весов = 1",
     abs(w_opt.sum() - 1) < 1e-4,
     f"{w_opt.sum():.6f}"),
    (f"Дюрация в [{H-H_TOL:.2f}, {H+H_TOL:.2f}] лет",
     abs(port_dur - H) <= H_TOL + 1e-4,
     f"{port_dur:.4f}"),
    (f"ОФЗ >= {W_MIN_OFZ*100:.0f}%",
     w_ofz_s >= W_MIN_OFZ - 1e-4,
     f"{w_ofz_s*100:.1f}%"),
    (f"Корпоратив <= {W_MAX_CORP*100:.0f}% (каждая)",
     all(w_opt[is_corp] <= W_MAX_CORP + 1e-4),
     ""),
    (f"Любая бумага <= {W_MAX*100:.0f}%",
     all(w_opt <= W_MAX + 1e-4),
     ""),
]
for label, flag, detail in checks:
    print(f"  {'✅' if flag else '❌'} {label:<38} {detail}")

# ============================================================
# БЛОК 4: СОХРАНЕНИЕ
# ============================================================

df[["Бумага", "ISIN", "тип", "вес_%", "ytm", "dur", "Ликвидность"]]\
    .rename(columns={
        "вес_%":       "Вес, %",
        "ytm":         "YTM, %",
        "dur":         "Дюрация, лет",
        "Ликвидность": "Ликвидность",
    })\
    .sort_values(["тип", "Вес, %"], ascending=[True, False])\
    .to_csv("portfolio_weights.csv", index=False, encoding="utf-8-sig")

print("\n✅ Сохранено: portfolio_weights.csv")
