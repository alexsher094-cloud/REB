#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа №4 — «ЭПР объектов различной формы»
========================================================
Скрипт реализует интерактивное приложение с графическим интерфейсом
на базе CustomTkinter + Matplotlib для расчёта и визуализации
эффективной площади рассеяния (ЭПР) восьми типов объектов:
  сфера, диск, прямоугольная пластина, конус,
  двугранный уголковый отражатель, трёхгранные отражатели,
  линза Люнеберга, решётка Ван-Атта.

Для запуска:
    python lab4_REB.py

Зависимости: numpy, matplotlib, customtkinter
"""

import sys
import numpy as np

# ── Конфигурация Matplotlib ДО импорта pyplot ──────────────────────
import matplotlib
matplotlib.use("TkAgg")

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt

import customtkinter as ctk

# ── Глобальные настройки Matplotlib ─────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})

# ══════════════════════════════════════════════════════════════════════
#  ТАБЛИЦА ВАРИАНТОВ
# ══════════════════════════════════════════════════════════════════════
# Формат: номер варианта → (f_ГГц, a_мм, b_мм, l_м)
VARIANTS = {
    1:  (2.0, 10, 21, 1.5),    2:  (2.5, 18, 22, 2.5),
    3:  (3.0, 23, 25, 3.5),    4:  (2.7,  5, 26, 4.0),
    5:  (1.5, 19, 29, 1.0),    6:  (1.0, 11, 25, 3.0),
    7:  (3.2, 13, 21, 2.0),    8:  (0.7, 12, 23, 1.5),
    9:  (0.9, 17, 29, 2.5),    10: (1.4, 19, 35, 1.5),
    11: (1.2, 16, 31, 2.5),    12: (2.6, 15, 33, 3.5),
    13: (1.3, 21, 37, 4.0),    14: (0.8,  8, 39, 1.0),
    15: (0.6,  7, 38, 3.0),    16: (2.1, 17, 30, 2.0),
}

C = 3.0e8  # скорость света, м/с


# ══════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════

def safe_sinc(x: np.ndarray) -> np.ndarray:
    """Безопасная sinc: sin(x)/x, равная 1 при x=0."""
    out = np.ones_like(x, dtype=float)
    mask = np.abs(x) > 1e-30
    out[mask] = np.sin(x[mask]) / x[mask]
    return out


def compute_params(variant: int) -> dict:
    """Вычисляет параметры варианта: f, λ, k, a, b, l, ka."""
    f_ghz, a_mm, b_mm, l_m = VARIANTS[variant]
    f = f_ghz * 1e9                        # Гц
    lam = C / f                             # м
    k = 2.0 * np.pi / lam                  # рад/м
    a = a_mm * 1e-3                         # м
    b = b_mm * 1e-3                         # м
    l = l_m                                 # м
    ka = k * a
    return {"f": f, "lam": lam, "k": k, "a": a, "b": b, "l": l, "ka": ka,
            "f_ghz": f_ghz, "a_mm": a_mm, "b_mm": b_mm, "l_m": l_m}


# ══════════════════════════════════════════════════════════════════════
#  ФУНКЦИИ ЗАДАЧ 1–8  (каждая возвращает matplotlib.figure.Figure)
# ══════════════════════════════════════════════════════════════════════

def task1_sphere(p: dict) -> Figure:
    """
    Задание 1 — ЭПР сферы (Рэлеевская область, a << λ).
    Формула (2): σ = 144·π⁵·a⁶ / λ⁴
    """
    lam = p["lam"]
    ratio = np.linspace(0.001, 0.1, 500)          # a/λ
    a_arr = ratio * lam                             # радиус сферы, м
    sigma = 144.0 * np.pi**5 * a_arr**6 / lam**4

    fig = Figure(figsize=(10, 4.5), dpi=100)
    fig.suptitle("Задание 1 — ЭПР сферы (Рэлеевская область, a << λ)", fontsize=13, fontweight="bold")

    ax1 = fig.add_subplot(121)
    ax1.plot(ratio, sigma, color="#2563eb", linewidth=1.8)
    ax1.set_xlabel("a / λ")
    ax1.set_ylabel("σ, м²")
    ax1.set_title("Линейный масштаб")

    ax2 = fig.add_subplot(122)
    ax2.semilogy(ratio, sigma, color="#dc2626", linewidth=1.8)
    ax2.set_xlabel("a / λ")
    ax2.set_ylabel("σ, м²")
    ax2.set_title("Логарифмический масштаб")

    fig.text(0.5, 0.01, "σ = 144π⁵a⁶ / λ⁴", ha="center", fontsize=12,
             style="italic", color="#444")
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    return fig


def task2_disk(p: dict) -> Figure:
    """
    Задание 2 — ЭПР диска (длинноволновая область, a << λ).
    σ_m = 4·π³·a⁴ / λ²
    σ(θ) = σ_m·(8ka / 3π)²·cos⁴(θ)
    """
    lam = p["lam"]; k = p["k"]; a = p["a"]
    sigma_m = 4.0 * np.pi**3 * a**4 / lam**2
    coeff = (8.0 * k * a / (3.0 * np.pi))**2

    theta = np.linspace(0, np.pi / 2, 500)
    sigma = sigma_m * coeff * np.cos(theta)**4
    sigma_db = 10.0 * np.log10(sigma + 1e-30)

    fig = Figure(figsize=(10, 4.5), dpi=100)
    fig.suptitle("Задание 2 — ЭПР диска (длинноволновая область, a << λ)", fontsize=12, fontweight="bold")

    ax1 = fig.add_subplot(121)
    ax1.plot(np.degrees(theta), sigma, color="#16a34a", linewidth=1.8)
    ax1.set_xlabel("θ, град")
    ax1.set_ylabel("σ, м²")
    ax1.set_title("Линейный масштаб")

    ax2 = fig.add_subplot(122)
    ax2.plot(np.degrees(theta), sigma_db, color="#9333ea", linewidth=1.8)
    ax2.set_xlabel("θ, град")
    ax2.set_ylabel("σ, дБ")
    ax2.set_title("Децибелы")

    fig.text(0.5, 0.01, "σ(θ) = σ_m·(8ka/3π)²·cos⁴θ,   σ_m = 4π³a⁴ / λ²",
             ha="center", fontsize=11, style="italic", color="#444")
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    return fig


def task3_plate(p: dict) -> Figure:
    """
    Задание 3 — ЭПР прямоугольной пластины (a,b >> λ).
    σ(θ,φ) = σ_m·[cos(θ)·sinc(2ka·sinθ·cosφ)·sinc(2kb·sinθ·sinφ)]²
    σ_m = 64·π·a²·b² / λ²
    """
    lam = p["lam"]; k = p["k"]
    a_plate = p["a"] * 100.0   # масштабируем для условия a,b >> λ
    b_plate = p["b"] * 100.0
    sigma_m = 64.0 * np.pi * a_plate**2 * b_plate**2 / lam**2

    theta = np.linspace(0.001, np.pi / 2 - 0.001, 200)
    phi = np.linspace(0, 2.0 * np.pi, 200)
    TH, PH = np.meshgrid(theta, phi)

    arg1 = 2.0 * k * a_plate * np.sin(TH) * np.cos(PH)
    arg2 = 2.0 * k * b_plate * np.sin(TH) * np.sin(PH)
    sigma_3d = sigma_m * (np.cos(TH) * safe_sinc(arg1) * safe_sinc(arg2))**2

    # Сечения
    phi_0 = 0.0      # xz
    phi_90 = np.pi / 2.0  # yz

    def _section(phi_val):
        arg1s = 2.0 * k * a_plate * np.sin(theta) * np.cos(phi_val)
        arg2s = 2.0 * k * b_plate * np.sin(theta) * np.sin(phi_val)
        return sigma_m * (np.cos(theta) * safe_sinc(arg1s) * safe_sinc(arg2s))**2

    sig_xz = _section(phi_0)
    sig_yz = _section(phi_90)

    fig = Figure(figsize=(12, 9), dpi=100)
    fig.suptitle(f"Задание 3 — ЭПР прямоугольной пластины  ({a_plate:.1f}×{b_plate:.1f} м)",
                 fontsize=12, fontweight="bold")

    # 3D поверхность — в дБ для наглядности гребешков
    sigma_3d_dB = 10.0 * np.log10(sigma_3d + 1e-30)

    ax1 = fig.add_subplot(221, projection="3d")
    surf = ax1.plot_surface(np.degrees(TH), np.degrees(PH),
                            sigma_3d_dB, cmap="jet", linewidth=0,
                            antialiased=True, rstride=2, cstride=2)
    fig.colorbar(surf, ax=ax1, shrink=0.5, pad=0.1, label="σ, дБм²")
    ax1.set_xlabel("θ, град"); ax1.set_ylabel("φ, град"); ax1.set_zlabel("σ, дБм²")
    ax1.set_title("3D: σ(θ, φ) в дБ")
    ax1.view_init(elev=30, azim=-60)

    # xz + yz сечения (линейный)
    ax2 = fig.add_subplot(222)
    ax2.plot(np.degrees(theta), sig_xz, color="#0ea5e9", linewidth=1.5, label="xz (φ=0°)")
    ax2.plot(np.degrees(theta), sig_yz, color="#f97316", linewidth=1.5, label="yz (φ=90°)")
    ax2.set_xlabel("θ, град"); ax2.set_ylabel("σ, м²")
    ax2.set_title("σ_m = 64πa²b² / λ²  (сечения)")
    ax2.legend(fontsize=9)

    # dB сечения
    ax3 = fig.add_subplot(223)
    ax3.plot(np.degrees(theta), 10.0 * np.log10(sig_xz + 1e-30),
             color="#0ea5e9", linewidth=1.5, label="xz (φ=0°)")
    ax3.plot(np.degrees(theta), 10.0 * np.log10(sig_yz + 1e-30),
             color="#f97316", linewidth=1.5, label="yz (φ=90°)")
    ax3.set_xlabel("θ, град"); ax3.set_ylabel("σ, дБ")
    ax3.set_title("Сечения в децибелах")
    ax3.set_ylim(None, 10.0 * np.log10(sigma_m + 1e-30) + 5)
    ax3.legend(fontsize=9)

    # Совмещённая диаграмма (полярная, нормированная)
    ax4 = fig.add_subplot(224, projection="polar")
    ax4.plot(theta, sig_xz / (sigma_m + 1e-30), color="#0ea5e9", linewidth=1.5, label="xz")
    ax4.plot(theta, sig_yz / (sigma_m + 1e-30), color="#f97316", linewidth=1.5, label="yz")
    ax4.set_title("Полярная диаграмма\n(нормированная)", fontsize=9, pad=15)
    ax4.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def task4_cone(p: dict) -> Figure:
    """
    Задание 4 — ЭПР конуса.
    σ = π·a²·tan²(α), α от 10° до 70°
    """
    a_cone = p["a"] * 10.0
    alpha = np.linspace(np.radians(10), np.radians(70), 500)
    sigma = np.pi * a_cone**2 * np.tan(alpha)**2
    sigma_db = 10.0 * np.log10(sigma + 1e-30)

    fig = Figure(figsize=(10, 4.5), dpi=100)
    fig.suptitle(f"Задание 4 — ЭПР конуса (a = {a_cone*1e3:.0f} мм)", fontsize=12, fontweight="bold")

    ax1 = fig.add_subplot(121)
    ax1.plot(np.degrees(alpha), sigma, color="#e11d48", linewidth=1.8)
    ax1.set_xlabel("α, град")
    ax1.set_ylabel("σ, м²")
    ax1.set_title("Линейный масштаб")

    ax2 = fig.add_subplot(122)
    ax2.plot(np.degrees(alpha), sigma_db, color="#7c3aed", linewidth=1.8)
    ax2.set_xlabel("α, град")
    ax2.set_ylabel("σ, дБ")
    ax2.set_title("Децибелы")

    fig.text(0.5, 0.01, "σ = πa²tg²α", ha="center", fontsize=12,
             style="italic", color="#444")
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    return fig


def task5_dihedral(p: dict) -> Figure:
    """
    Задание 5 — ЭПР двугранного уголкового отражателя.
    σ_m = 8·π·a²·b² / λ²
    (a) σ(φ): PO+PTD — сумма комплексных амплитуд поля:
        E_res = E_двукр.(φ) + E_краев1(φ) + E_краев2(φ)
        σ = σ_m·|E_res|² / |E_res(0)|²
    (b) σ(θ) = σ_m·sinθ·sinc²(kb·sinθ), θ ∈ [0.001°, 90°]
    """
    lam = p["lam"]; k = p["k"]; a = p["a"]; b = p["b"]
    sigma_m = 8.0 * np.pi * a**2 * b**2 / lam**2

    # (a) зависимость от φ — комплексная амплитуда поля (PO + PTD)
    # Диапазон с отступом от ±45° для избежания сингулярности краевых волн
    phi = np.linspace(-np.pi / 4 + 0.01, np.pi / 4 - 0.01, 500)

    # PO: поле двукратного переотражения (физическая оптика)
    f_PO = np.cos(2.0 * phi) * safe_sinc(2.0 * k * a * np.sin(phi))

    # PTD: краевые дифракционные волны (Уфимцев)
    # Каждая краевая волна имеет дифракционный коэффициент ~ 1/sqrt(2πka)
    # и фазовый сдвиг e^{-jπ/4}
    edge_coeff = np.exp(-1j * np.pi / 4.0) / np.sqrt(2.0 * np.pi * k * a + 1e-30)
    f_E1 = edge_coeff * np.cos(2.0 * phi) / np.sin(np.pi / 4.0 - phi)
    f_E2 = edge_coeff * np.cos(2.0 * phi) / np.sin(np.pi / 4.0 + phi)

    # Результирующее комплексное поле
    E_res = f_PO + f_E1 + f_E2

    # Нормировка: σ(φ=0) = σ_m
    E_res_0 = f_PO[0] + f_E1[0] + f_E2[0]
    sigma_phi = sigma_m * np.abs(E_res)**2 / (np.abs(E_res_0)**2 + 1e-30)

    # (b) зависимость от θ
    theta = np.linspace(np.radians(0.001), np.pi / 2, 500)
    sigma_theta = sigma_m * np.sin(theta) * safe_sinc(k * b * np.sin(theta))**2

    fig = Figure(figsize=(10, 8), dpi=100)
    fig.suptitle("Задание 5 — Двугранный уголковый отражатель", fontsize=13, fontweight="bold")

    ax1 = fig.add_subplot(221)
    ax1.plot(np.degrees(phi), sigma_phi, color="#0d9488", linewidth=1.5)
    ax1.set_xlabel("φ, град"); ax1.set_ylabel("σ, м²")
    ax1.set_title("(a) Плоскость ⊥ ребру (линейный масштаб)")

    ax2 = fig.add_subplot(222)
    ax2.plot(np.degrees(phi), 10.0 * np.log10(sigma_phi + 1e-30), color="#be185d", linewidth=1.5)
    ax2.set_xlabel("φ, град"); ax2.set_ylabel("σ, дБ")
    ax2.set_title("(a) Плоскость ⊥ ребру (децибелы)")

    ax3 = fig.add_subplot(223)
    ax3.plot(np.degrees(theta), sigma_theta, color="#0d9488", linewidth=1.5)
    ax3.set_xlabel("θ, град"); ax3.set_ylabel("σ, м²")
    ax3.set_title("(b) Ортогональная плоскость (линейный масштаб)")

    ax4 = fig.add_subplot(224)
    ax4.plot(np.degrees(theta), 10.0 * np.log10(sigma_theta + 1e-30), color="#be185d", linewidth=1.5)
    ax4.set_xlabel("θ, град"); ax4.set_ylabel("σ, дБ")
    ax4.set_title("(b) Ортогональная плоскость (децибелы)")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def task6_trihedral(p: dict) -> Figure:
    """
    Задание 6 — Максимальные ЭПР трёхгранных отражателей.
    Треугольный: σ = (4/3)·π·a⁴ / λ²
    Прямоугольный: σ = 12·π·a⁴ / λ²
    Круглый:       σ = 2·π·a⁴ / λ²
    a от λ до 20λ.
    """
    lam = p["lam"]
    a_arr = np.linspace(lam, 20.0 * lam, 500)

    sig_tri = (4.0 / 3.0) * np.pi * a_arr**4 / lam**2
    sig_rect = 12.0 * np.pi * a_arr**4 / lam**2
    sig_circ = 2.0 * np.pi * a_arr**4 / lam**2

    ratio = a_arr / lam

    fig = Figure(figsize=(10, 5), dpi=100)
    fig.suptitle("Задание 6 — Максимальные ЭПР трёхгранных отражателей", fontsize=12, fontweight="bold")

    ax1 = fig.add_subplot(121)
    ax1.plot(ratio, sig_tri,  color="#2563eb", linewidth=1.5,
             label="Треуг.: σ = (4/3)πa⁴/λ²")
    ax1.plot(ratio, sig_rect, color="#dc2626", linewidth=1.5,
             label="Прямоуг.: σ = 12πa⁴/λ²")
    ax1.plot(ratio, sig_circ, color="#16a34a", linewidth=1.5,
             label="Кругл.: σ = 2πa⁴/λ²")
    ax1.set_xlabel("a / λ"); ax1.set_ylabel("σ, м²")
    ax1.set_title("Линейный масштаб"); ax1.legend(fontsize=8, loc="upper left")

    ax2 = fig.add_subplot(122)
    ax2.plot(ratio, 10.0 * np.log10(sig_tri + 1e-30),  color="#2563eb", linewidth=1.5,
             label="Треуг.: σ = (4/3)πa⁴/λ²")
    ax2.plot(ratio, 10.0 * np.log10(sig_rect + 1e-30), color="#dc2626", linewidth=1.5,
             label="Прямоуг.: σ = 12πa⁴/λ²")
    ax2.plot(ratio, 10.0 * np.log10(sig_circ + 1e-30), color="#16a34a", linewidth=1.5,
             label="Кругл.: σ = 2πa⁴/λ²")
    ax2.set_xlabel("a / λ"); ax2.set_ylabel("σ, дБ")
    ax2.set_title("Децибелы"); ax2.legend(fontsize=8, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def task7_luneburg(p: dict) -> Figure:
    """
    Задание 7 — Линза Люнеберга с кольцом.
    R = 20λ, l от 0.1R до 1.8R.
    σ = (4π/λ²)·(2Rl − l²)²
    """
    lam = p["lam"]
    R = 20.0 * lam
    l_arr = np.linspace(0.1 * R, 1.8 * R, 500)
    sigma = (4.0 * np.pi / lam**2) * (2.0 * R * l_arr - l_arr**2)**2

    # Найдём максимум
    idx_max = np.argmax(sigma)
    l_max = l_arr[idx_max]
    sig_max = sigma[idx_max]

    sigma_db = 10.0 * np.log10(sigma + 1e-30)

    fig = Figure(figsize=(10, 4.5), dpi=100)
    fig.suptitle(f"Задание 7 — Линза Люнеберга (R = 20λ = {R:.2f} м)", fontsize=12, fontweight="bold")

    ax1 = fig.add_subplot(121)
    ax1.plot(l_arr / R, sigma, color="#0891b2", linewidth=1.8)
    ax1.plot(l_max / R, sig_max, "ro", markersize=8, label=f"Максимум: l={l_max/R:.2f}R")
    ax1.set_xlabel("l / R"); ax1.set_ylabel("σ, м²")
    ax1.set_title("Линейный масштаб"); ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(122)
    ax2.plot(l_arr / R, sigma_db, color="#a21caf", linewidth=1.8)
    ax2.plot(l_max / R, sigma_db[idx_max], "ro", markersize=8, label=f"Максимум: {sigma_db[idx_max]:.1f} дБ")
    ax2.set_xlabel("l / R"); ax2.set_ylabel("σ, дБ")
    ax2.set_title("Децибелы"); ax2.legend(fontsize=9)

    fig.text(0.5, 0.01, "σ = (4π/λ²)·(2Rl − l²)²", ha="center", fontsize=12,
             style="italic", color="#444")
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    return fig


def task8_van_atta(p: dict) -> Figure:
    """
    Задание 8 — Решётка Ван-Атта.
    N = 64·variant, S = N·λ²/4
    σ(θ) = (4πS²/λ²)·sin⁴((π/2)·cosθ), θ от 0° до 80°
    """
    lam = p["lam"]; variant = p.get("_variant", 1)
    N = 64 * variant
    S = N * lam**2 / 4.0

    theta = np.linspace(0, np.radians(80), 500)
    sigma = (4.0 * np.pi * S**2 / lam**2) * np.sin((np.pi / 2.0) * np.cos(theta))**4
    sigma_db = 10.0 * np.log10(sigma + 1e-30)

    fig = Figure(figsize=(10, 4.5), dpi=100)
    fig.suptitle(f"Задание 8 — Решётка Ван-Атта (N = {N})", fontsize=12, fontweight="bold")

    ax1 = fig.add_subplot(121)
    ax1.plot(np.degrees(theta), sigma, color="#ca8a04", linewidth=1.8)
    ax1.set_xlabel("θ, град"); ax1.set_ylabel("σ, м²")
    ax1.set_title("Линейный масштаб")

    ax2 = fig.add_subplot(122)
    ax2.plot(np.degrees(theta), sigma_db, color="#b45309", linewidth=1.8)
    ax2.set_xlabel("θ, град"); ax2.set_ylabel("σ, дБ")
    ax2.set_title("Децибелы")

    fig.text(0.5, 0.01, "σ = (4πS²/λ²)·sin⁴((π/2)cosθ)", ha="center",
             fontsize=12, style="italic", color="#444")

    # Аналитическая проверка: σ_max = πλ²N²/4 при θ=0
    sigma_max_anal = np.pi * lam**2 * N**2 / 4.0
    sigma_max_graph = sigma[0]
    fig.text(0.5, 0.06,
             f"Аналитич. проверка: σ_max = πλ²N²/4 = {sigma_max_anal:.4e} м²  |  "
             f"График при θ=0: {sigma_max_graph:.4e} м²",
             ha="center", fontsize=10, color="#555",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                       edgecolor="#cccccc"))

    fig.tight_layout(rect=[0, 0.10, 1, 0.92])
    return fig


# ══════════════════════════════════════════════════════════════════════
#  ОПИСАНИЯ ЗАДАЧ (для кнопок и лога)
# ══════════════════════════════════════════════════════════════════════
TASK_NAMES = [
    "1. Сфера (Рэлеев)",
    "2. Диск (длинноволн.)",
    "3. Прямоуг. пластина",
    "4. Конус",
    "5. Двугран. отражатель",
    "6. Трёхгран. отражатели",
    "7. Линза Люнеберга",
    "8. Решётка Ван-Атта",
]

TASK_FUNCS = [task1_sphere, task2_disk, task3_plate, task4_cone,
              task5_dihedral, task6_trihedral, task7_luneburg, task8_van_atta]


# ══════════════════════════════════════════════════════════════════════
#  ГЛАВНОЕ ПРИЛОЖЕНИЕ
# ══════════════════════════════════════════════════════════════════════

class App(ctk.CTk):
    """Главное окно приложения."""

    def __init__(self):
        super().__init__()
        self.title("Лаб. №4 — ЭПР объектов различной формы")
        self.geometry("1400x820")
        self.minsize(1000, 600)

        # Текущий вариант
        self._variant = 1
        # Сохранённые figures для «Сохранить все» (по одному на задание)
        self._saved_figs: dict[int, tuple[str, Figure]] = {}

        # ── Сетка основного окна ────────────────────────────────────
        self.grid_columnconfigure(0, weight=0, minsize=310)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ── Левая боковая панель ────────────────────────────────────
        self._build_sidebar()

        # ── Основная область (matplotlib) ────────────────────────────
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Показать приветствие
        self._show_welcome()

    # ────────────────────────────────────────────────────────────────
    def _build_sidebar(self):
        """Создаёт левую боковую панель с элементами управления."""
        sidebar = ctk.CTkFrame(self, width=310, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)          # фиксированная ширина
        sidebar.grid_rowconfigure(99, weight=1)  # «пружина» для лога

        row = 0

        # ── Заголовок ──────────────────────────────────────────────
        ctk.CTkLabel(sidebar, text="Лабораторная работа №4",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     anchor="w").grid(row=row, column=0, columnspan=2,
                                      sticky="w", padx=12, pady=(14, 0))
        row += 1
        ctk.CTkLabel(sidebar, text="ЭПР объектов различной формы",
                     font=ctk.CTkFont(size=12, slant="italic"),
                     text_color="gray", anchor="w"
                     ).grid(row=row, column=0, columnspan=2,
                            sticky="w", padx=12)
        row += 1

        # Разделитель
        ctk.CTkFrame(sidebar, height=2, fg_color="#d1d5db"
                     ).grid(row=row, column=0, columnspan=2,
                            sticky="ew", padx=12, pady=10)
        row += 1

        # ── Выбор варианта ─────────────────────────────────────────
        ctk.CTkLabel(sidebar, text="Вариант:", font=ctk.CTkFont(size=13, weight="bold"),
                     anchor="w").grid(row=row, column=0, sticky="w", padx=12)
        row += 1

        self.variant_combo = ctk.CTkOptionMenu(
            sidebar,
            values=[str(v) for v in range(1, 17)],
            command=self._on_variant_change,
            width=120,
        )
        self.variant_combo.set("1")
        self.variant_combo.grid(row=row, column=0, sticky="w", padx=12)
        row += 1

        # Разделитель
        ctk.CTkFrame(sidebar, height=2, fg_color="#d1d5db"
                     ).grid(row=row, column=0, columnspan=2,
                            sticky="ew", padx=12, pady=10)
        row += 1

        # ── Таблица параметров ─────────────────────────────────────
        ctk.CTkLabel(sidebar, text="Параметры:", font=ctk.CTkFont(size=13, weight="bold"),
                     anchor="w").grid(row=row, column=0, columnspan=2,
                                      sticky="w", padx=12)
        row += 1

        self._param_labels = {}
        param_names = ["f", "λ", "k", "a", "b", "l", "ka"]
        for pname in param_names:
            lbl = ctk.CTkLabel(sidebar, text=f"  {pname} = —", anchor="w",
                               font=ctk.CTkFont(family="Courier New", size=12))
            lbl.grid(row=row, column=0, columnspan=2, sticky="w", padx=12)
            self._param_labels[pname] = lbl
            row += 1

        self._update_params()

        # Разделитель
        ctk.CTkFrame(sidebar, height=2, fg_color="#d1d5db"
                     ).grid(row=row, column=0, columnspan=2,
                            sticky="ew", padx=12, pady=10)
        row += 1

        # ── Кнопки заданий ─────────────────────────────────────────
        ctk.CTkLabel(sidebar, text="Задания:", font=ctk.CTkFont(size=13, weight="bold"),
                     anchor="w").grid(row=row, column=0, columnspan=2,
                                      sticky="w", padx=12)
        row += 1

        self._task_buttons = []
        colors = ["#2563eb", "#16a34a", "#0ea5e9", "#e11d48",
                  "#0d9488", "#7c3aed", "#0891b2", "#ca8a04"]
        for i, name in enumerate(TASK_NAMES):
            btn = ctk.CTkButton(
                sidebar, text=name, width=260,
                fg_color=colors[i % len(colors)],
                hover_color=colors[i % len(colors)],
                anchor="w",
                command=lambda idx=i: self._run_task(idx),
            )
            btn.grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=2)
            self._task_buttons.append(btn)
            row += 1

        # ── Кнопка «Сохранить все» ─────────────────────────────────
        ctk.CTkButton(
            sidebar, text="💾  Сохранить все", width=260,
            fg_color="#475569", hover_color="#334155",
            command=self._save_all,
        ).grid(row=row, column=0, columnspan=2, sticky="ew", padx=12, pady=(12, 4))
        row += 1

        # Разделитель
        ctk.CTkFrame(sidebar, height=2, fg_color="#d1d5db"
                     ).grid(row=row, column=0, columnspan=2,
                            sticky="ew", padx=12, pady=6)
        row += 1

        # ── Лог ────────────────────────────────────────────────────
        ctk.CTkLabel(sidebar, text="Журнал:", font=ctk.CTkFont(size=12, weight="bold"),
                     anchor="w").grid(row=row, column=0, columnspan=2,
                                      sticky="w", padx=12)
        row += 1

        self.log_box = ctk.CTkTextbox(sidebar, width=270, height=140,
                                      font=ctk.CTkFont(family="Courier New", size=10))
        self.log_box.grid(row=row, column=0, columnspan=2,
                          sticky="nsew", padx=12, pady=(0, 12))
        self.log_box.grid_rowconfigure(99, weight=1)
        # Лог занимает всё свободное место (row 99 = «пружина»)

    # ────────────────────────────────────────────────────────────────
    def _update_params(self):
        """Обновляет отображение параметров текущего варианта."""
        p = compute_params(self._variant)
        self._param_labels["f"].configure(
            text=f"  f   = {p['f_ghz']:.1f} ГГц  ({p['f']:.3e} Гц)")
        self._param_labels["λ"].configure(
            text=f"  λ   = {p['lam'] * 100:.2f} см  ({p['lam']:.4f} м)")
        self._param_labels["k"].configure(
            text=f"  k   = {p['k']:.4f} рад/м")
        self._param_labels["a"].configure(
            text=f"  a   = {p['a_mm']} мм  ({p['a']:.4f} м)")
        self._param_labels["b"].configure(
            text=f"  b   = {p['b_mm']} мм  ({p['b']:.4f} м)")
        self._param_labels["l"].configure(
            text=f"  l   = {p['l_m']:.1f} м")
        self._param_labels["ka"].configure(
            text=f"  ka  = {p['ka']:.4f}")

    # ────────────────────────────────────────────────────────────────
    def _on_variant_change(self, value):
        """Обработчик смены варианта."""
        self._variant = int(value)
        self._update_params()
        self._log(f"Выбран вариант {self._variant}")

    # ────────────────────────────────────────────────────────────────
    def _log(self, msg: str):
        """Добавляет запись в журнал."""
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")

    # ────────────────────────────────────────────────────────────────
    def _embed(self, fig: Figure) -> FigureCanvasTkAgg:
        """Встраивает Figure в главную область, возвращает canvas."""
        for w in self.main_frame.winfo_children():
            w.destroy()

        # Промежуточный tk.Frame, чтобы NavigationToolbar2Tk
        # мог использовать pack() внутри себя без конфликта с grid
        import tkinter as tk
        inner = tk.Frame(self.main_frame)
        inner.grid(row=0, column=0, sticky="nsew")

        canvas = FigureCanvasTkAgg(fig, master=inner)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, inner)
        toolbar.update()
        return canvas

    # ────────────────────────────────────────────────────────────────
    def _show_welcome(self):
        """Отображает приветственную фигуру."""
        fig = Figure(figsize=(8, 6), dpi=100)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(0.5, 0.6,
                "Лабораторная работа №4\n\n"
                "«ЭПР объектов различной формы»\n\n"
                "Выберите вариант и нажмите кнопку задания\n"
                "для построения соответствующего графика.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=16, fontweight="bold", color="#1e293b",
                linespacing=1.6)
        ax.text(0.5, 0.15,
                "Задания 1–8  |  Варианты 1–16",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="#64748b")
        self._embed(fig)
        self._log("Приложение запущено. Выберите задание.")

    # ────────────────────────────────────────────────────────────────
    def _run_task(self, idx: int):
        """Выполняет задание idx (0–7)."""
        name = TASK_NAMES[idx]
        self._log(f"→ Вычисление: {name}")
        self._variant  # доступен через замыкание
        p = compute_params(self._variant)
        p["_variant"] = self._variant  # для задачи 8

        try:
            fig = TASK_FUNCS[idx](p)
            self._embed(fig)
            self._saved_figs[idx] = (name, fig)  # перезаписываем повторный клик
            self._log(f"  ✓ {name} — готово")
        except Exception as e:
            self._log(f"  ✗ Ошибка: {e}")
            import traceback
            traceback.print_exc()

    # ────────────────────────────────────────────────────────────────
    def _save_all(self):
        """Сохраняет все построенные графики в PNG-файлы."""
        if not self._saved_figs:
            self._log("Нет графиков для сохранения.")
            return
        saved_count = 0
        for idx in sorted(self._saved_figs):
            name, fig = self._saved_figs[idx]
            # Сформировать имя файла из названия задачи
            safe_name = name.split(".")[0].strip().replace(" ", "_")
            fname = f"lab4_var{self._variant}_{safe_name}.png"
            try:
                fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
                self._log(f"  💾 Сохранено: {fname}")
                saved_count += 1
            except Exception as e:
                self._log(f"  ✗ Ошибка сохранения {fname}: {e}")
        self._log(f"Сохранено {saved_count} файл(ов).")


# ══════════════════════════════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ctk.set_appearance_mode("System")   # «System» / «Dark» / «Light»
    ctk.set_default_color_theme("blue")  # «blue» / «green» / «dark-blue»

    app = App()
    app.mainloop()
