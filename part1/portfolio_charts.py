
# ============================================================
#  ВИЗУАЛИЗАЦИЯ ОБЛИГАЦИОННОГО ПОРТФЕЛЯ
#  Входной файл: portfolio_weights.csv
#  Выходные файлы: portfolio_weights_bar.png
#                  portfolio_ytm_contrib.png
#                  portfolio_map.png
# ============================================================

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json

# ------------------------------------------------------------
# БЛОК 0: ЗАГРУЗКА ДАННЫХ
# ------------------------------------------------------------

df = pd.read_csv("portfolio_weights.csv", encoding="utf-8-sig")
df = df[df["Вес, %"] > 0.1].copy()
df["Имя"] = df["Бумага"].apply(lambda s: " ".join(str(s).split()[:2]))

COLOR = {"ОФЗ": "#4f9cf4", "Корпоратив": "#f4a24f"}
df["color"] = df["тип"].map(COLOR)

port_ytm = np.dot(df["Вес, %"] / 100, df["YTM, %"])
port_dur = np.dot(df["Вес, %"] / 100, df["Дюрация, лет"])
w_ofz    = df.loc[df["тип"] == "ОФЗ",        "Вес, %"].sum()
w_corp   = df.loc[df["тип"] == "Корпоратив", "Вес, %"].sum()

# ============================================================
# CHART 1: Горизонтальный бар — веса бумаг
# ============================================================

active = df.sort_values("Вес, %")

fig1 = go.Figure()
for typ, clr in COLOR.items():
    d = active[active["тип"] == typ]
    fig1.add_trace(go.Bar(
        x=d["Вес, %"],
        y=d["Имя"],
        orientation="h",
        name=typ,
        marker_color=clr,
        text=[f"  {w:.1f}%  |  YTM {y:.1f}%"
              for w, y in zip(d["Вес, %"], d["YTM, %"])],
        textposition="outside",
        textfont=dict(size=11),
        cliponaxis=False,
    ))

fig1.update_layout(
    title={"text": (
        "Структура портфеля по бумагам<br>"
        f"<span style='font-size:14px;font-weight:normal;'>"
        f"YTM: {port_ytm:.2f}%  |  Дюрация: {port_dur:.2f} л.  |  "
        f"ОФЗ {w_ofz:.0f}% / Корп {w_corp:.0f}%</span>"
    )},
    barmode="stack",
    legend=dict(orientation="h", yanchor="bottom", y=1.14,
                xanchor="center", x=0.5),
    margin=dict(t=130, r=160, b=60, l=160),
    xaxis=dict(range=[0, active["Вес, %"].max() * 1.6]),
)
fig1.update_xaxes(title_text="Вес, %")
fig1.update_yaxes(tickfont=dict(size=12))

fig1.write_image("portfolio_weights_bar.png")
with open("portfolio_weights_bar.png.meta.json", "w") as f:
    json.dump({"caption": "Структура портфеля: вес и YTM каждой бумаги"}, f)
print("✅ portfolio_weights_bar.png")

# ============================================================
# CHART 2: Горизонтальный бар — вклад каждой бумаги в YTM
# ============================================================

df["вклад"] = df["Вес, %"] / 100 * df["YTM, %"]
contrib = df.sort_values("вклад")

fig2 = go.Figure()
for typ, clr in COLOR.items():
    d = contrib[contrib["тип"] == typ]
    fig2.add_trace(go.Bar(
        x=d["вклад"],
        y=d["Имя"],
        orientation="h",
        name=typ,
        marker_color=clr,
        text=[f"  {v:.2f} п.п." for v in d["вклад"]],
        textposition="outside",
        textfont=dict(size=11),
        cliponaxis=False,
    ))

# Вертикаль итогового YTM — без перекрытия через add_annotation
fig2.add_vline(x=port_ytm, line_dash="dot", line_color="#888", line_width=2)
fig2.add_annotation(
    x=port_ytm, y=1.04, xref="x", yref="paper",
    text=f"YTM = {port_ytm:.2f}%",
    showarrow=False, font=dict(size=12, color="#555"),
    xanchor="left", bgcolor="rgba(255,255,255,0.85)", borderpad=3
)

fig2.update_layout(
    title={"text": (
        "Вклад каждой бумаги в YTM портфеля<br>"
        f"<span style='font-size:14px;font-weight:normal;'>"
        f"Итого: {port_ytm:.2f}%</span>"
    )},
    barmode="stack",
    legend=dict(orientation="h", yanchor="bottom", y=1.14,
                xanchor="center", x=0.5),
    margin=dict(t=130, r=160, b=60, l=160),
    xaxis=dict(range=[0, contrib["вклад"].max() * 1.65]),
)
fig2.update_xaxes(title_text="Вклад в YTM, п.п.")
fig2.update_yaxes(tickfont=dict(size=12))

fig2.write_image("portfolio_ytm_contrib.png")
with open("portfolio_ytm_contrib.png.meta.json", "w") as f:
    json.dump({"caption": "Вклад бумаг в итоговую доходность портфеля"}, f)
print("✅ portfolio_ytm_contrib.png")

# ============================================================
# CHART 3: Пузырьковый scatter — дюрация vs YTM
#
# Ключевой приём против наезда текста:
#   mode="markers" — текст убран из трейсов
#   add_annotation() для каждой точки с ручными ax/ay смещениями
# ============================================================

# ax = сдвиг вправо/влево в пикселях от точки
# ay = сдвиг вниз/вверх (положительный = вниз, отрицательный = вверх)
OFFSETS = {
    "ВТБ Б-1-245":              (-10, -26),
    "Банк ВТБ":                 (-10, -26),
    "АФК Система,":             ( 12, -26),
    "МТС, 002Р-07":             (-55, -26),
    "Роснефть, 002P-04":        ( -8,  28),
    "МБЭС, 002P-04":            ( 12,  30),
    "РЖД, 001P-41R":            ( 12, -26),
    "Россети, 001P-04R":        ( 12, -26),
    "Атомэнергопром, 001P-10":  ( 12,  30),
    "Россия, 26228":            (-10,  30),
    "Россия, 26241":            (-62, -26),
    "Россия, 26245":            ( 12, -26),
}

fig3 = go.Figure()

for typ, clr in COLOR.items():
    d = df[df["тип"] == typ]
    fig3.add_trace(go.Scatter(
        x=d["Дюрация, лет"],
        y=d["YTM, %"],
        name=typ,
        mode="markers",
        marker=dict(
            size=d["Вес, %"] * 3.8,
            color=clr,
            opacity=0.85,
            line=dict(width=1.5, color="white"),
        ),
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            "Дюрация: %{x:.2f} л.<br>"
            "YTM: %{y:.2f}%<extra></extra>"
        ),
        customdata=d["Имя"],
    ))

# Аннотации с индивидуальными смещениями — гарантированно без наезда
for _, row in df.iterrows():
    ax, ay = OFFSETS.get(row["Имя"], (12, -26))
    fig3.add_annotation(
        x=row["Дюрация, лет"],
        y=row["YTM, %"],
        text=row["Имя"],
        showarrow=True,
        arrowhead=0,
        arrowwidth=1,
        arrowcolor="#ccc",
        ax=ax,
        ay=ay,
        font=dict(size=10),
        bgcolor="rgba(255,255,255,0.80)",
        borderpad=2,
        xanchor="center",
    )

# Вертикаль целевой дюрации
fig3.add_vline(x=port_dur, line_dash="dot", line_color="#888", line_width=2)
fig3.add_annotation(
    x=port_dur, y=0.97, xref="x", yref="paper",
    text=f"D = {port_dur:.2f} л.",
    showarrow=False, font=dict(size=12, color="#555"),
    xanchor="left", bgcolor="rgba(255,255,255,0.85)", borderpad=3
)

fig3.update_layout(
    title={"text": (
        "Карта портфеля: дюрация vs доходность<br>"
        "<span style='font-size:14px;font-weight:normal;'>"
        "Размер кружка = вес бумаги в портфеле</span>"
    )},
    legend=dict(orientation="h", yanchor="bottom", y=1.14,
                xanchor="center", x=0.5),
    margin=dict(t=130, r=60, b=70, l=70),
)
fig3.update_xaxes(
    title_text="Дюрация, лет",
    range=[-0.3, df["Дюрация, лет"].max() * 1.1]
)
fig3.update_yaxes(
    title_text="YTM, %",
    range=[df["YTM, %"].min() - 2.5, df["YTM, %"].max() + 4]
)

fig3.write_image("portfolio_map.png")
with open("portfolio_map.png.meta.json", "w") as f:
    json.dump({"caption": "Карта риска: дюрация vs доходность"}, f)
print("✅ portfolio_map.png")
