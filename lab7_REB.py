#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторная работа №7
«DSSS — Расширение спектра методом прямой последовательности»

Полностью автономный скрипт с GUI на CustomTkinter.
Зависимости: numpy, matplotlib, customtkinter.

Авторы: (ваши ФИО)
Группа: (номер группы)
"""

import sys
import os

# ---------------------------------------------------------------
# Обязательно устанавливаем TkAgg-бэкенд ДО импорта pyplot
# ---------------------------------------------------------------
import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# CustomTkinter должен быть установлен: pip install customtkinter
import customtkinter as ctk
from scipy.signal import butter, sosfiltfilt

# ===============================================================
# Настройки Matplotlib
# ===============================================================
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

# ===============================================================
# Таблица вариантов: вариант → (bits1, SF1, bits2, SF2)
# PG = SF (линейный), PG_dB = 10·log10(SF)
# ===============================================================
VARIANTS = {
    1:  (5, 5, 50, 25),      2:  (10, 10, 55, 30),
    3:  (15, 15, 60, 35),     4:  (20, 20, 65, 40),
    5:  (25, 25, 70, 5),      6:  (30, 30, 75, 10),
    7:  (35, 35, 80, 15),     8:  (40, 40, 85, 20),
    9:  (45, 5, 90, 25),      10: (50, 10, 95, 30),
    11: (55, 15, 100, 35),    12: (60, 20, 5, 40),
    13: (65, 25, 10, 5),      14: (70, 30, 15, 10),
    15: (75, 35, 20, 15),     16: (80, 40, 25, 20),
    17: (85, 5, 30, 25),      18: (90, 10, 35, 30),
    19: (95, 15, 40, 35),     20: (100, 20, 45, 40),
}


# ===============================================================
# Генератор М-последовательности (РСЛОС / LFSR)
# ===============================================================
def _generate_m_sequence(min_length: int) -> tuple[np.ndarray, int]:
    """
    Генерация М-последовательности максимальной длины на основе
    регистра сдвига с линейной обратной связью (РСЛОС / LFSR).

    Параметры
    ----------
    min_length : int
        Минимальная требуемая длина ПСП.

    Возвращает
    --------
    tuple[np.ndarray, int]
        (последовательность {0,1} длины >= min_length, степень n полинома)
    """
    # Примитивные полиномы для разных степеней n (период = 2^n - 1)
    primitive_polys = {
        2: [2, 1], 3: [3, 2], 4: [4, 3], 5: [5, 3],
        6: [6, 5], 7: [7, 6], 8: [8, 6, 5, 4],
        9: [9, 5], 10: [10, 7], 11: [11, 9],
        12: [12, 11, 10, 4], 13: [13, 12, 11, 8],
        14: [14, 13, 12, 2], 15: [15, 14],
    }

    # Минимальное n: 2^n - 1 >= min_length
    n = 2
    while (2**n - 1) < min_length:
        n += 1

    taps = primitive_polys.get(n, [n, n - 1])

    # LFSR: начальное состояние — все единицы
    reg = [1] * n
    seq = []
    period = 2**n - 1
    for _ in range(period):
        seq.append(reg[-1])
        fb = 0
        for t in taps:
            fb ^= reg[t - 1]
        reg = [fb] + reg[:-1]

    return np.array(seq[:min_length]), n


# ===============================================================
# Функция моделирования DSSS
# ===============================================================
def lab7_dsss(N_bits: int, SF: int, spc: int = 20, fc_cycles: int = 5):
    """
    Полная цепочка моделирования DSSS.

    Параметры
    ---------
    N_bits : int
        Количество информационных бит.
    SF : int
        Коэффициент расширения (чипов на бит).
    spc : int
        Отсчётов на чип (samples per chip).
    fc_cycles : int
        Количество периодов несущей на один чип.

    Возвращает
    --------
    tuple
        (fig, N_bits, SF, PG_dB, data_bits, recovered_bits)
        fig — объект matplotlib Figure с 8 графиками (7 + АКФ ПСП).
    """
    # Начальное значение ГСЧ — привязано к параметрам набора
    np.random.seed(N_bits * 100 + SF)

    # Общие параметры
    N_chips = N_bits * SF                        # всего чипов
    N_samp = N_chips * spc                       # всего отсчётов
    PG = SF                                       # выигрыш обработки (линейн.)
    PG_dB = 10.0 * np.log10(SF)                  # выигрыш обработки (дБ)
    fc = fc_cycles                                # частота несущей (цикл/чип)

    # Временная ось в единицах чипов
    t_chip = np.arange(N_samp, dtype=float) / spc

    # -----------------------------------------------------------
    # 1. Генерация информационных бит и преобразование в биполярный
    # -----------------------------------------------------------
    data_bits = np.random.randint(0, 2, size=N_bits)      # {0, 1}
    data_bip = 2.0 * data_bits - 1.0                       # {-1, +1}

    # -----------------------------------------------------------
    # 2. Генерация PN-последовательности (М-последовательность, РСЛОС)
    # -----------------------------------------------------------
    pn_bits, lfsr_n = _generate_m_sequence(SF)           # SF чипов {0,1}
    pn_bip = 2.0 * pn_bits - 1.0                           # {-1, +1}
    lfsr_period = 2**lfsr_n - 1

    # -----------------------------------------------------------
    # 3. Расширение спектра (Spreading)
    #    Каждый бит умножается на PN-последовательность
    # -----------------------------------------------------------
    spread_seq = np.kron(data_bip, pn_bip)                 # N_chips значений

    # -----------------------------------------------------------
    # 4. Формирование непрерывных сигналов
    # -----------------------------------------------------------
    # Информационный сигнал (ступенчатый, без расширения)
    data_cont = np.repeat(data_bip, spc * SF)

    # Расширенный сигнал (ступенчатый)
    spread_cont = np.repeat(spread_seq, spc)

    # PN-последовательность (повторяем для всех бит)
    pn_pattern = np.tile(pn_bip, N_bits)
    pn_cont = np.repeat(pn_pattern, spc)

    # -----------------------------------------------------------
    # 5. Несущая
    # -----------------------------------------------------------
    carrier = np.cos(2.0 * np.pi * fc * t_chip)

    # -----------------------------------------------------------
    # 6. BPSK-модуляция
    # -----------------------------------------------------------
    bpsk_sig = spread_cont * carrier

    # -----------------------------------------------------------
    # 7. Спектр BPSK-сигнала (FFT)
    # -----------------------------------------------------------
    bpsk_fft = np.fft.fft(bpsk_sig)
    bpsk_freq = np.fft.fftfreq(N_samp, d=1.0 / (spc * fc))
    bpsk_psd = np.abs(bpsk_fft) ** 2 / N_samp
    # Переставим оси: отрицательные → нулевая → положительные частоты
    bpsk_freq_shifted = np.fft.fftshift(bpsk_freq)
    bpsk_psd_shifted = np.fft.fftshift(bpsk_psd)
    bpsk_psd_db = 10.0 * np.log10(bpsk_psd_shifted + 1e-15)

    # -----------------------------------------------------------
    # 8. Демодуляция: умножение на несущую + ФНЧ (Баттерворт)
    # -----------------------------------------------------------
    demod_raw = bpsk_sig * carrier  # синхронное детектирование

    # ФНЧ Баттерворта 6-го порядка
    # Частота среза: 2/spc от Найквиста (в норм. единицах)
    # Пропускает полосу до ~1 цикл/чип, подавляет 2fc компоненту
    nyq = spc * fc / 2.0                       # Найквист, цикл/чип
    cutoff_norm = min(fc / nyq, 0.99)          # нормированная частота среза
    sos = butter(6, cutoff_norm, btype='low', output='sos')
    demod_filt = sosfiltfilt(sos, demod_raw)

    # -----------------------------------------------------------
    # 9. Дешифрование (Despreading): умножение на PN + интегрирование
    # -----------------------------------------------------------
    despread_raw = demod_filt * pn_cont

    # Интегрирование по периоду бита (spc*SF отсчётов)
    # В идеале (без шума, идеальная синхронизация) даёт прямоугольный
    # сигнал: mean(despread_raw за битовый период) = ±0.5
    window = spc * SF
    compressed_bit_val = np.array([
        np.mean(despread_raw[i * window:(i + 1) * window])
        for i in range(N_bits)
    ])
    # Формируем ступенчатый (прямоугольный) сигнал — разбиваем на биты
    compressed = np.repeat(compressed_bit_val, window)

    # -----------------------------------------------------------
    # 10. Спектр демодулированного сигнала
    # -----------------------------------------------------------
    demod_fft = np.fft.fft(demod_filt)
    demod_freq = np.fft.fftfreq(N_samp, d=1.0 / (spc * fc))
    demod_psd = np.abs(demod_fft) ** 2 / N_samp
    demod_freq_shifted = np.fft.fftshift(demod_freq)
    demod_psd_shifted = np.fft.fftshift(demod_psd)
    demod_psd_db = 10.0 * np.log10(demod_psd_shifted + 1e-15)
    # Нормализация к 0 дБ (максимум)
    demod_psd_db_norm = demod_psd_db - np.max(demod_psd_db)

    # -----------------------------------------------------------
    # 11. Восстановление бит
    # -----------------------------------------------------------
    # Выборка в центре каждого бита
    recovered_bip = compressed_bit_val.copy()
    recovered_bits = (recovered_bip > 0).astype(int)

    # ===========================================================
    # ПОСТРОЕНИЕ ГРАФИКОВ
    # ===========================================================
    # Определяем, сколько бит показывать
    if N_bits <= 10:
        show_bits = N_bits
    else:
        show_bits = 8

    show_samp = show_bits * spc * SF  # сколько отсчётов показывать
    t_show = t_chip[:show_samp]

    # Figure с gridspec: 6 строк × 2 столбца
    fig = Figure(figsize=(16, 18), dpi=100)
    gs = gridspec.GridSpec(
        6, 2,
        height_ratios=[1, 1, 1.2, 1, 0.9, 1.2],
        hspace=0.55, wspace=0.30,
        left=0.06, right=0.97, top=0.96, bottom=0.03,
    )

    # Цвета
    col_data = "#1a73e8"
    col_spread = "#e63946"
    col_bpsk = "#2d6a4f"
    col_demod = "#457b9d"
    col_compressed = "#6a4c93"
    col_expected = "#e76f51"

    # ==========================================================
    # График 1: Информационный сигнал (биполярный)
    # ==========================================================
    ax1 = fig.add_subplot(gs[0, 0])
    data_show = data_cont[:show_samp]
    ax1.step(t_show, data_show, where="post", linewidth=1.2, color=col_data)
    # Отметки битовых значений (красные точки в центре каждого бита)
    for i in range(show_bits):
        center_idx = int((i + 0.5) * spc * SF)
        if center_idx < show_samp:
            ax1.plot(
                t_chip[center_idx], data_cont[center_idx],
                "ro", markersize=7, zorder=5,
            )
            # Числовое значение бита
            ax1.annotate(
                str(data_bits[i]),
                (t_chip[center_idx], data_cont[center_idx]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
                color="red",
                fontweight="bold",
            )
    ax1.set_title(f"1. Информационный сигнал ({show_bits}/{N_bits} бит)",
                  fontsize=11, fontweight="bold")
    ax1.set_xlabel("Время (чипы)")
    ax1.set_ylabel("Амплитуда")
    ax1.set_ylim(-1.5, 1.5)
    ax1.axhline(y=0, color="black", linewidth=0.5)

    # ==========================================================
    # График 2: Расширенный сигнал (после spreading)
    # ==========================================================
    ax2 = fig.add_subplot(gs[0, 1])
    spread_show = spread_cont[:show_samp]
    ax2.step(t_show, spread_show, where="post", linewidth=1.0, color=col_spread)
    # Вертикальные пунктирные линии на границах бит
    for i in range(show_bits + 1):
        boundary = i * SF
        if boundary <= show_samp / spc:
            ax2.axvline(x=boundary, color="gray", linestyle="--", linewidth=0.7, alpha=0.6)
    ax2.set_title(f"2. Расширенный сигнал SF={SF} ({show_bits}/{N_bits} бит)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("Время (чипы)")
    ax2.set_ylabel("Амплитуда")
    ax2.set_ylim(-1.5, 1.5)
    ax2.axhline(y=0, color="black", linewidth=0.5)

    # ==========================================================
    # График 3: BPSK-сигнал (на всю ширину)
    # ==========================================================
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(t_show, bpsk_sig[:show_samp], linewidth=0.6, color=col_bpsk)
    # Тонкие вертикальные линии на границах бит
    for i in range(show_bits + 1):
        boundary = i * SF
        if boundary <= show_samp / spc:
            ax3.axvline(x=boundary, color="lightgray", linestyle=":", linewidth=0.5)
    ax3.set_title(f"3. BPSK-сигнал (fc={fc} цикл/чип) — {show_bits}/{N_bits} бит",
                  fontsize=11, fontweight="bold")
    ax3.set_xlabel("Время (чипы)")
    ax3.set_ylabel("Амплитуда")
    ax3.set_ylim(-1.5, 1.5)

    # ==========================================================
    # График 4: Спектр BPSK-сигнала (на всю ширину)
    # ==========================================================
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(bpsk_freq_shifted, bpsk_psd_db, linewidth=0.8, color="#333333")
    # Отметка несущей частоты
    ax4.axvline(x=fc, color="red", linestyle="--", linewidth=1.2, label=f"fc={fc}")
    ax4.axvline(x=-fc, color="red", linestyle="--", linewidth=1.2)
    # Отметки ±1 чип-скорость (границы главного лепестка BPSK)
    chip_rate = fc  # 1/T_chip ≈ fc в наших единицах
    ax4.axvline(x=fc + chip_rate, color="blue", linestyle=":", linewidth=1.0,
                label="±1/T_chip")
    ax4.axvline(x=fc - chip_rate, color="blue", linestyle=":", linewidth=1.0)
    ax4.axvline(x=-fc + chip_rate, color="blue", linestyle=":", linewidth=1.0)
    ax4.axvline(x=-fc - chip_rate, color="blue", linestyle=":", linewidth=1.0)
    ax4.set_title("4. Спектр BPSK-сигнала (мощность, дБ)",
                  fontsize=11, fontweight="bold")
    ax4.set_xlabel("Частота (цикл/чип)")
    ax4.set_ylabel("PSD (дБ)")
    ax4.set_xlim(-3 * fc, 3 * fc)
    ax4.legend(fontsize=9, loc="upper right")

    # ==========================================================
    # График 5: Демодулированный сигнал (ФНЧ)
    # ==========================================================
    ax5 = fig.add_subplot(gs[3, 0])
    demod_show = demod_filt[:show_samp]
    expected_spread = spread_cont[:show_samp] / 2.0  # ожидаемый = spread/2
    ax5.plot(t_show, demod_show, linewidth=0.8, color=col_demod, label="Демодулированный")
    ax5.plot(t_show, expected_spread, linewidth=1.2, linestyle="--",
             color=col_expected, label="Ожидаемый (spread/2)")
    ax5.set_title("5. Демодулированный сигнал (после ФНЧ)",
                  fontsize=11, fontweight="bold")
    ax5.set_xlabel("Время (чипы)")
    ax5.set_ylabel("Амплитуда")
    ax5.legend(fontsize=8, loc="upper right")

    # ==========================================================
    # График 6: Сжатый сигнал (после дешифрования/интегрирования)
    # ==========================================================
    ax6 = fig.add_subplot(gs[3, 1])
    compressed_show = compressed[:show_samp]
    expected_data = data_cont[:show_samp] / 2.0  # ожидаемый = data/2
    ax6.step(t_show, compressed_show, where="post", linewidth=1.2, color=col_compressed,
             label="Сжатый (интегрированный)")
    ax6.step(t_show, expected_data, where="post", linewidth=1.2, linestyle="--",
             color=col_expected, label="Ожидаемый (data/2)")
    # Отметки восстановленных бит
    for i in range(show_bits):
        center_idx = int((i + 0.5) * spc * SF)
        if center_idx < show_samp:
            marker = "go" if recovered_bits[i] == data_bits[i] else "rx"
            ax6.plot(t_chip[center_idx], compressed[center_idx],
                     marker, markersize=7, zorder=5)
    ax6.set_title("6. Сжатый сигнал (после дешифрования)",
                  fontsize=11, fontweight="bold")
    ax6.set_xlabel("Время (чипы)")
    ax6.set_ylabel("Амплитуда")
    ax6.legend(fontsize=8, loc="upper right")

    # ==========================================================
    # График 7: Спектр демодулированного сигнала (на всю ширину)
    # ==========================================================
    ax7 = fig.add_subplot(gs[5, :])
    ax7.plot(demod_freq_shifted, demod_psd_db_norm, linewidth=0.8, color="#555555")
    # Отметки ±1/T_chip (границы полосы расширенного сигнала)
    ax7.axvline(x=chip_rate, color="blue", linestyle="--", linewidth=1.0,
                label="±1/T_chip")
    ax7.axvline(x=-chip_rate, color="blue", linestyle="--", linewidth=1.0)
    # Отметки ±1/T_bit (ширина полосы узкополосного сигнала)
    bit_rate = fc / SF
    ax7.axvline(x=bit_rate, color="red", linestyle="--", linewidth=1.0,
                label=f"±1/T_bit (±{bit_rate:.2f})")
    ax7.axvline(x=-bit_rate, color="red", linestyle="--", linewidth=1.0)
    ax7.set_title("7. Спектр демодулированного сигнала (норм., дБ)",
                  fontsize=11, fontweight="bold")
    ax7.set_xlabel("Частота (цикл/чип)")
    ax7.set_ylabel("PSD (дБ, норм.)")
    ax7.set_xlim(-3 * fc, 3 * fc)
    ax7.legend(fontsize=9, loc="upper right")

    # ==========================================================
    # График 8: АКФ ПСП (М-последовательности)
    # ==========================================================
    ax8 = fig.add_subplot(gs[4, :])

    # Нормированная автокорреляционная функция ПСП
    acf_full = np.correlate(pn_bip, pn_bip, mode='full')
    acf_norm = acf_full / acf_full[len(acf_full) // 2]  # нормировка к 1
    lags = np.arange(-(len(pn_bip) - 1), len(pn_bip))

    ax8.stem(lags, acf_norm, linefmt='#0d9488', markerfmt='o', basefmt='#cccccc')
    ax8.set_title(f"8. АКФ ПСП (М-последовательность, n={lfsr_n}, период={lfsr_period})",
                  fontsize=11, fontweight="bold")
    ax8.set_xlabel("Лаг (чипы)")
    ax8.set_ylabel("R(τ) / R(0)")
    ax8.axhline(y=-1.0 / lfsr_period, color='red', linestyle='--', linewidth=0.8,
                label=f'Уровень бок. лепестка ≈ −1/{lfsr_period}')
    ax8.legend(fontsize=9, loc='upper right')

    # Общий заголовок
    fig.suptitle(
        f"DSSS — N_bits={N_bits}, SF={SF}, PG={PG_dB:.1f} дБ, "
        f"fc={fc} цикл/чип, spc={spc}",
        fontsize=13, fontweight="bold", y=0.99,
    )

    return fig, N_bits, SF, PG_dB, data_bits, recovered_bits


# ===============================================================
# GUI-приложение
# ===============================================================
class App(ctk.CTk):
    """Главное окно приложения ЛР7 DSSS."""

    def __init__(self):
        super().__init__()

        # ---- Настройка окна ----
        self.title("ТОРЭБ — ЛР7: DSSS — Расширение спектра")
        self.geometry("1500x920")
        self.minsize(1100, 700)

        # ---- Сетка корневого окна ----
        self.grid_columnconfigure(0, weight=0, minsize=300)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---- Левая панель (sidebar) ----
        self._build_sidebar()

        # ---- Основная область (main) ----
        self._build_main()

        # ---- Начальное состояние ----
        self._show_welcome()

        # Храним последнюю фигуру для сохранения
        self._last_fig = None

    # -----------------------------------------------------------
    # Построение боковой панели
    # -----------------------------------------------------------
    def _build_sidebar(self):
        """Создаёт левую панель с элементами управления."""
        sidebar = ctk.CTkFrame(self, width=300, corner_radius=8)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        sidebar.grid_propagate(False)  # фиксированная ширина

        # Разрешаем внутреннюю прокрутку
        sidebar.grid_rowconfigure(0, weight=0)  # заголовок
        sidebar.grid_rowconfigure(1, weight=0)  # вариант
        sidebar.grid_rowconfigure(2, weight=1)  # параметры
        sidebar.grid_rowconfigure(3, weight=0)  # кнопки
        sidebar.grid_rowconfigure(4, weight=0)  # кнопка сохранения
        sidebar.grid_rowconfigure(5, weight=0)  # лог
        sidebar.grid_columnconfigure(0, weight=1)

        # ---- Заголовок ----
        hdr = ctk.CTkFrame(sidebar, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=10, pady=(15, 5))

        ctk.CTkLabel(
            hdr, text="Лабораторная работа №7",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            hdr,
            text="Расширение спектра (DSSS)",
            font=ctk.CTkFont(size=12, slant="italic"),
            text_color="gray",
        ).pack(anchor="w", pady=(2, 0))

        # ---- Разделитель ----
        ctk.CTkFrame(sidebar, height=2, fg_color="#444").grid(
            row=0, column=0, sticky="ew", padx=10, pady=(50, 0)
        )

        # ---- Выбор варианта ----
        var_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        var_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            var_frame, text="Вариант:", font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w")

        self._variant_var = ctk.StringVar(value="1")
        variant_values = [str(i) for i in range(1, 21)]
        self._variant_combo = ctk.CTkOptionMenu(
            var_frame,
            variable=self._variant_var,
            values=variant_values,
            command=self._on_variant_changed,
            width=260,
        )
        self._variant_combo.pack(anchor="w", pady=(4, 0))

        # ---- Параметры наборов ----
        params_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        params_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(10, 5))

        ctk.CTkLabel(
            params_frame,
            text="Параметры наборов:",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w")

        # Набор 1
        set1_box = ctk.CTkFrame(params_frame, corner_radius=6, border_width=1)
        set1_box.pack(fill="x", pady=(6, 4))
        self._set1_label = ctk.CTkLabel(
            set1_box,
            text="",
            font=ctk.CTkFont(size=11, family="Courier New"),
            justify="left",
        )
        self._set1_label.pack(anchor="w", padx=8, pady=6)

        # Набор 2
        set2_box = ctk.CTkFrame(params_frame, corner_radius=6, border_width=1)
        set2_box.pack(fill="x", pady=(0, 4))
        self._set2_label = ctk.CTkLabel(
            set2_box,
            text="",
            font=ctk.CTkFont(size=11, family="Courier New"),
            justify="left",
        )
        self._set2_label.pack(anchor="w", padx=8, pady=6)

        # Обновим параметры для варианта 1
        self._update_params_display()

        # ---- Кнопки вычисления ----
        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(8, 4))

        self._btn_set1 = ctk.CTkButton(
            btn_frame,
            text="▶ Набор 1 (короткий)",
            command=self._run_set1,
            width=260,
            height=36,
            corner_radius=6,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#2d6a4f",
            hover_color="#40916c",
        )
        self._btn_set1.pack(pady=(4, 4))

        self._btn_set2 = ctk.CTkButton(
            btn_frame,
            text="▶ Набор 2 (длинный)",
            command=self._run_set2,
            width=260,
            height=36,
            corner_radius=6,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#6a4c93",
            hover_color="#8b6db0",
        )
        self._btn_set2.pack(pady=(0, 4))

        # ---- Кнопка сохранения ----
        save_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        save_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(4, 4))

        self._btn_save = ctk.CTkButton(
            save_frame,
            text="💾 Сохранить оба набора",
            command=self._save_both,
            width=260,
            height=32,
            corner_radius=6,
            font=ctk.CTkFont(size=11),
            fg_color="#555",
            hover_color="#777",
        )
        self._btn_save.pack(pady=(2, 2))

        # ---- Лог ----
        log_frame = ctk.CTkFrame(sidebar, corner_radius=6)
        log_frame.grid(row=5, column=0, sticky="ew", padx=10, pady=(8, 10))

        ctk.CTkLabel(
            log_frame,
            text="Журнал:",
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(anchor="w", padx=8, pady=(6, 2))

        self._log_text = ctk.CTkTextbox(
            log_frame,
            height=140,
            width=260,
            font=ctk.CTkFont(size=10, family="Courier New"),
            corner_radius=4,
        )
        self._log_text.pack(fill="x", padx=8, pady=(0, 8))
        self._log("Приложение запущено.")
        self._log("Выберите вариант и нажмите кнопку.")

    # -----------------------------------------------------------
    # Построение основной области
    # -----------------------------------------------------------
    def _build_main(self):
        """Создаёт основную область для отображения графиков."""
        self._main_frame = ctk.CTkFrame(self, corner_radius=8)
        self._main_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 6), pady=6)

        self._main_frame.grid_rowconfigure(0, weight=1)
        self._main_frame.grid_columnconfigure(0, weight=1)

    # -----------------------------------------------------------
    # Вспомогательные методы
    # -----------------------------------------------------------
    def _log(self, msg: str):
        """Добавляет строку в журнал."""
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")

    def _get_variant_params(self):
        """Возвращает параметры текущего варианта."""
        v = int(self._variant_var.get())
        bits1, sf1, bits2, sf2 = VARIANTS[v]
        return v, bits1, sf1, bits2, sf2

    def _on_variant_changed(self, value):
        """Обработчик смены варианта."""
        self._update_params_display()
        v, b1, s1, b2, s2 = self._get_variant_params()
        self._log(f"Выбран вариант {v}")

    def _update_params_display(self):
        """Обновляет отображение параметров наборов."""
        v, b1, s1, b2, s2 = self._get_variant_params()
        pg1_db = 10.0 * np.log10(s1) if s1 > 0 else 0
        pg2_db = 10.0 * np.log10(s2) if s2 > 0 else 0

        self._set1_label.configure(
            text=(
                f"Набор 1 (короткий):\n"
                f"  Бит: {b1:>3d}  SF: {s1:>2d}\n"
                f"  PG: {pg1_db:.1f} дБ"
            )
        )
        self._set2_label.configure(
            text=(
                f"Набор 2 (длинный):\n"
                f"  Бит: {b2:>3d}  SF: {s2:>2d}\n"
                f"  PG: {pg2_db:.1f} дБ"
            )
        )

    def _embed(self, parent, fig):
        """Встраивает matplotlib figure в CTkFrame."""
        # Очищаем предыдущие виджеты
        for w in parent.winfo_children():
            w.destroy()

        # Промежуточный tk.Frame, чтобы NavigationToolbar2Tk
        # мог использовать pack() внутри себя без конфликта с grid
        import tkinter as tk
        inner = tk.Frame(parent)
        inner.pack(fill="both", expand=True)

        canvas = FigureCanvasTkAgg(fig, master=inner)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        toolbar = NavigationToolbar2Tk(canvas, inner)
        toolbar.update()
        return canvas

    def _show_welcome(self):
        """Показывает приветственное сообщение в основной области."""
        # Создаём простую фигуру с приветствием
        fig = Figure(figsize=(14, 10), facecolor="white")
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

        ax.text(
            5, 7,
            "Лабораторная работа №7",
            fontsize=28, ha="center", va="center",
            fontweight="bold", color="#333333",
        )
        ax.text(
            5, 5.8,
            "DSSS — Расширение спектра\nметодом прямой последовательности",
            fontsize=18, ha="center", va="center",
            color="#555555",
        )
        ax.text(
            5, 4.0,
            "Выберите вариант в боковой панели\n"
            "и нажмите кнопку для построения графиков",
            fontsize=14, ha="center", va="center",
            color="#888888",
        )
        ax.text(
            5, 2.2,
            "Набор 1 — короткая последовательность (малый SF)\n"
            "Набор 2 — длинная последовательность (большой SF)",
            fontsize=12, ha="center", va="center",
            color="#aaaaaa",
        )

        # Декоративные элементы
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch(
            (1.5, 1.0), 7, 7,
            boxstyle="round,pad=0.3",
            linewidth=2,
            edgecolor="#cccccc",
            facecolor="#f8f8f8",
        )
        ax.add_patch(box)

        self._embed(self._main_frame, fig)

    def _run_computation(self, N_bits, SF, set_name):
        """Запускает вычисления и отображает результат."""
        self._log(f"--- Вычисление: {set_name} ---")
        self._log(f"  N_bits={N_bits}, SF={SF}")

        # Блокируем кнопки
        self._btn_set1.configure(state="disabled")
        self._btn_set2.configure(state="disabled")
        self.update_idletasks()

        try:
            self._log(f"  Выполняется моделирование DSSS...")
            fig, nb, sf, pg_db, data_bits, rec_bits = lab7_dsss(N_bits, SF)

            self._log(f"  Готово! PG = {pg_db:.1f} дБ")

            # Показываем график
            self._embed(self._main_frame, fig)
            self._last_fig = fig

            self._log(f"  {set_name}: отображено.")

            # Вывод сравнения бит (первые 20)
            n_compare = min(20, N_bits)
            match = np.sum(data_bits[:n_compare] == rec_bits[:n_compare])
            self._log(f"  Совпадение бит (первые {n_compare}): {match}/{n_compare}")

        except Exception as e:
            self._log(f"  ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Разблокируем кнопки
            self._btn_set1.configure(state="normal")
            self._btn_set2.configure(state="normal")

    def _run_set1(self):
        """Обработчик кнопки 'Набор 1'."""
        v, b1, s1, b2, s2 = self._get_variant_params()
        self._run_computation(b1, s1, f"Набор 1 (вариант {v})")

    def _run_set2(self):
        """Обработчик кнопки 'Набор 2'."""
        v, b1, s1, b2, s2 = self._get_variant_params()
        self._run_computation(b2, s2, f"Набор 2 (вариант {v})")

    def _save_both(self):
        """Сохраняет оба набора графиков в PNG-файлы."""
        v, b1, s1, b2, s2 = self._get_variant_params()
        self._log("=== Сохранение обоих наборов ===")
        self._btn_save.configure(state="disabled")
        self.update_idletasks()

        try:
            # Набор 1
            self._log(f"  Генерация Набора 1 (bits={b1}, SF={s1})...")
            fig1, _, _, _, _, _ = lab7_dsss(b1, s1)
            fname1 = f"lab7_var{v}_set1_bits{b1}_SF{s1}.png"
            fig1.savefig(fname1, dpi=150, bbox_inches="tight", facecolor="white")
            self._log(f"  Сохранено: {fname1}")
            plt.close(fig1)

            # Набор 2
            self._log(f"  Генерация Набора 2 (bits={b2}, SF={s2})...")
            fig2, _, _, _, _, _ = lab7_dsss(b2, s2)
            fname2 = f"lab7_var{v}_set2_bits{b2}_SF{s2}.png"
            fig2.savefig(fname2, dpi=150, bbox_inches="tight", facecolor="white")
            self._log(f"  Сохранено: {fname2}")
            plt.close(fig2)

            self._log("  Оба набора успешно сохранены!")

        except Exception as e:
            self._log(f"  ОШИБКА при сохранении: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._btn_save.configure(state="normal")


# ===============================================================
# Точка входа
# ===============================================================
def main():
    # Настройка внешнего вида CustomTkinter
    ctk.set_appearance_mode("light")       # "light", "dark", "system"
    ctk.set_default_color_theme("blue")    # "blue", "green", "dark-blue"

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
