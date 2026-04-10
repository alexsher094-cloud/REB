import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.fft import fft, fftshift
from scipy.signal import correlate
import sys
import warnings
warnings.filterwarnings('ignore')

# Настройка внешнего вида CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ==================== ОБЩИЕ ПАРАМЕТРЫ ====================
IF = 50e6           # Промежуточная частота, Гц
Fs = 200e6          # Частота дискретизации, Гц
Ts = 1 / Fs         # Период дискретизации, с
Ta = 100e-6         # Время анализа, с (увеличено для пачки ЛЧМ)
c0 = 3e8            # Скорость света, м/с
lambda_c = 0.03     # Длина волны, м (для f=10 ГГц)
f0 = 10e9           # Рабочая частота РЛС, Гц

# Вектор времени
t = np.arange(0, Ta, Ts)
Nt = len(t)
f = np.arange(Nt) * Fs / Nt

# ==================== ПАРАМЕТРЫ СИГНАЛОВ (глобальные переменные) ====================
# Прямоугольный импульс
PULSE_Tu = 0.25e-6
PULSE_Tpr = 1.5e-6
PULSE_N_pt = 1

# Сигнал с ЛЧМ
LFM_Tu = 1e-6
LFM_Tpr = 5e-6
LFM_deltaF = 20e6
LFM_N_pt = 1

# Код Баркера
BARKER_CODE = [1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1]
BARKER_Tu = 0.25e-6

# М-последовательность
MSEQ_CODE = [1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1]
MSEQ_Tu = 0.25e-6

# Пачка прямоугольных импульсов
BURST_Tu = 0.25e-6
BURST_Tpr = 1.5e-6
BURST_N_pt = 4

# Пачка ЛЧМ сигналов
LFM_BURST_Tu = 2e-6           # Длительность одного ЛЧМ импульса
LFM_BURST_Tpr = 5e-6          # Период повторения
LFM_BURST_N_pt = 4            # Количество импульсов в пачке
LFM_BURST_deltaF = 10e6       # Девиация частоты для каждого импульса

# ==================== ФУНКЦИЯ НЕОПРЕДЕЛЕННОСТИ ====================
def ambiguity_function_full(U, t, Ts, tau_min, tau_max, N_tau, f_max_display):
    tau_vec = np.linspace(tau_min, tau_max, N_tau)
    Nt = len(U)
    max_shift = int(np.ceil(max(abs(tau_min), abs(tau_max)) / Ts))
    N_pad = Nt + 2 * max_shift
    U_padded = np.pad(U, (max_shift, max_shift), 'constant')
    amf = np.zeros((N_tau, N_pad), dtype=complex)
    
    for i, tau in enumerate(tau_vec):
        shift_samples = int(np.round(tau / Ts))
        if shift_samples >= 0:
            U_shifted = np.roll(U_padded, shift_samples)
            U_shifted[:shift_samples] = 0
        else:
            U_shifted = np.roll(U_padded, shift_samples)
            U_shifted[shift_samples:] = 0
        product = U_padded * np.conj(U_shifted)
        amf[i, :] = fft(product)
    
    f_full = np.arange(N_pad) * Fs / N_pad
    f_full = f_full - f_full[N_pad//2]
    amf = fftshift(amf, axes=1)
    idx_f = np.abs(f_full) <= f_max_display
    f_plot = f_full[idx_f] / 1e6
    amf_cut = amf[:, idx_f]
    amf_abs = np.abs(amf_cut)
    if np.max(amf_abs) > 0:
        amf_abs = amf_abs / np.max(amf_abs)
    return amf_abs, tau_vec, f_plot

# ==================== ФУНКЦИИ ДЛЯ ГЕНЕРАЦИИ ВСЕХ СИГНАЛОВ ====================

def generate_pulse_signal():
    """Генерация прямоугольного импульса"""
    S_ampl = np.zeros(Nt, dtype=float)
    for n in range(PULSE_N_pt):
        mask = (t >= n * PULSE_Tpr) & (t < n * PULSE_Tpr + PULSE_Tu)
        S_ampl[mask] = 1.0
    
    U = S_ampl * np.exp(1j * 2 * np.pi * IF * t)
    return U

def generate_lfm_signal():
    """Генерация сигнала с ЛЧМ"""
    S_ampl = np.zeros(Nt, dtype=float)
    for n in range(LFM_N_pt):
        mask = (t >= n * LFM_Tpr) & (t < n * LFM_Tpr + LFM_Tu)
        S_ampl[mask] = 1.0
    
    U = np.zeros(Nt, dtype=complex)
    for n in range(LFM_N_pt):
        mask = (t >= n * LFM_Tpr) & (t < n * LFM_Tpr + LFM_Tu)
        t_local = t[mask] - n * LFM_Tpr
        phase = 2 * np.pi * IF * t_local + 2 * np.pi * (LFM_deltaF/(2*LFM_Tu)) * t_local**2
        U[mask] = S_ampl[mask] * np.exp(1j * phase)
    
    return U

def generate_barker_signal():
    """Генерация кода Баркера"""
    S_ampl = np.zeros(Nt, dtype=float)
    for i, val in enumerate(BARKER_CODE):
        mask = (t >= i * BARKER_Tu) & (t < i * BARKER_Tu + BARKER_Tu)
        S_ampl[mask] = val
    
    U = S_ampl * np.exp(1j * 2 * np.pi * IF * t)
    return U

def generate_msequence_signal():
    """Генерация М-последовательности"""
    S_ampl = np.zeros(Nt, dtype=float)
    for i, val in enumerate(MSEQ_CODE):
        mask = (t >= i * MSEQ_Tu) & (t < i * MSEQ_Tu + MSEQ_Tu)
        S_ampl[mask] = val
    
    U = S_ampl * np.exp(1j * 2 * np.pi * IF * t)
    return U

def generate_burst_signal():
    """Генерация пачки прямоугольных импульсов"""
    S_ampl = np.zeros(Nt, dtype=float)
    for n in range(BURST_N_pt):
        mask = (t >= n * BURST_Tpr) & (t < n * BURST_Tpr + BURST_Tu)
        S_ampl[mask] = 1.0
    
    U = S_ampl * np.exp(1j * 2 * np.pi * IF * t)
    return U

def generate_lfm_burst_signal():
    """Генерация пачки ЛЧМ сигналов"""
    S_ampl = np.zeros(Nt, dtype=float)
    for n in range(LFM_BURST_N_pt):
        mask = (t >= n * LFM_BURST_Tpr) & (t < n * LFM_BURST_Tpr + LFM_BURST_Tu)
        S_ampl[mask] = 1.0
    
    U = np.zeros(Nt, dtype=complex)
    for n in range(LFM_BURST_N_pt):
        mask = (t >= n * LFM_BURST_Tpr) & (t < n * LFM_BURST_Tpr + LFM_BURST_Tu)
        t_local = t[mask] - n * LFM_BURST_Tpr
        phase = 2 * np.pi * IF * t_local + 2 * np.pi * (LFM_BURST_deltaF/(2*LFM_BURST_Tu)) * t_local**2
        U[mask] = S_ampl[mask] * np.exp(1j * phase)
    
    return U

# ==================== ГЛАВНОЕ ПРИЛОЖЕНИЕ ====================
class RadarSignalApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Радиолокационные сигналы - Анализ функции неопределенности")
        self.root.geometry("1600x900")
        
        # Настройка сетки
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Создание информационной панели
        self.setup_info_panel()
        
        # Создание прокручиваемого фрейма для графиков
        self.scrollable_frame = ctk.CTkScrollableFrame(self.root, corner_radius=10)
        self.scrollable_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Заголовок
        title_label = ctk.CTkLabel(self.scrollable_frame, text=" АНАЛИЗ ФУНКЦИИ НЕОПРЕДЕЛЕННОСТИ РАДИОЛОКАЦИОННЫХ СИГНАЛОВ", 
                                    font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=10)
        
        # Фрейм для размещения всех графиков
        self.plots_frame = ctk.CTkFrame(self.scrollable_frame)
        self.plots_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Генерация и отображение всех графиков
        self.generate_all_plots()
        
    def setup_info_panel(self):
        """Правая панель с информацией по заданиям"""
        self.info_frame = ctk.CTkFrame(self.root, width=400, corner_radius=10)
        self.info_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.info_frame.grid_propagate(False)
        
        # Заголовок
        title_label = ctk.CTkLabel(self.info_frame, text=" ИНФОРМАЦИЯ ПО ЗАДАНИЯМ", 
                                    font=ctk.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=10)
        
        # Создание прокручиваемого фрейма для информации
        self.info_scrollable = ctk.CTkScrollableFrame(self.info_frame, height=800)
        self.info_scrollable.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Словарь для хранения фреймов информации по каждому заданию
        self.info_frames = {}
        
        # Создание информационных блоков с динамическими данными
        self.create_info_blocks()
        
        # Добавление общей информации
        self.add_general_info()
        
    def create_info_blocks(self):
        """Создание информационных блоков с данными из переменных"""
        
        # Задание 1: Прямоугольный импульс
        delta_R_pulse = c0 * PULSE_Tu / 2
        delta_V_pulse = (1/PULSE_Tu) * lambda_c / 2
        spectrum_width = 1/PULSE_Tu/1e6
        
        self.create_info_block(
            task_num=1,
            title="Задание 1: Прямоугольный импульс",
            signal_type="Прямоугольный радиоимпульс",
            params={
                "Длительность импульса (τи)": f"{PULSE_Tu*1e6:.2f} мкс",
                "Период повторения (Tпр)": f"{PULSE_Tpr*1e6:.2f} мкс",
                "Количество импульсов": str(PULSE_N_pt),
                "Промежуточная частота (IF)": f"{IF/1e6:.0f} МГц",
                "Частота дискретизации": f"{Fs/1e6:.0f} МГц"
            },
            resolutions={
                "Разрешение по дальности (ΔR)": f"{delta_R_pulse:.1f} м",
                "Разрешение по скорости (ΔV)": f"{delta_V_pulse:.2f} м/с",
                "Ширина спектра": f"~{spectrum_width:.1f} МГц",
                "База сигнала": "1"
            }
        )
        
        # Задание 2: Сигнал с ЛЧМ
        delta_R_lfm = c0 / (2 * LFM_deltaF)
        delta_V_lfm = (1/LFM_Tu) * lambda_c / 2
        base = LFM_Tu * LFM_deltaF
        
        self.create_info_block(
            task_num=2,
            title="Задание 2: Сигнал с ЛЧМ",
            signal_type="Линейно-частотно-модулированный сигнал",
            params={
                "Длительность импульса (τи)": f"{LFM_Tu*1e6:.2f} мкс",
                "Период повторения (Tпр)": f"{LFM_Tpr*1e6:.2f} мкс",
                "Девиация частоты (Δf)": f"{LFM_deltaF/1e6:.0f} МГц",
                "Количество импульсов": str(LFM_N_pt),
                "Промежуточная частота (IF)": f"{IF/1e6:.0f} МГц"
            },
            resolutions={
                "Разрешение по дальности (ΔR)": f"{delta_R_lfm:.1f} м",
                "Разрешение по скорости (ΔV)": f"{delta_V_lfm:.2f} м/с",
                "База сигнала": f"{base:.0f}",
                "Коэффициент сжатия": f"{base:.0f}"
            }
        )
        
        # Задание 3: Код Баркера
        code_str = ''.join(['1' if x == 1 else '-1' for x in BARKER_CODE])
        delta_R_barker = c0 * BARKER_Tu / 2
        delta_V_barker = (1/(len(BARKER_CODE)*BARKER_Tu)) * lambda_c / 2
        side_lobe_level = 1/len(BARKER_CODE)
        side_lobe_db = 20*np.log10(side_lobe_level)
        
        self.create_info_block(
            task_num=3,
            title="Задание 3: Код Баркера",
            signal_type="Фазоманипулированный сигнал (код Баркера)",
            params={
                "Длина кода": str(len(BARKER_CODE)),
                "Кодовая последовательность": code_str,
                "Длительность элементарного импульса": f"{BARKER_Tu*1e6:.2f} мкс",
                "Общая длительность сигнала": f"{len(BARKER_CODE) * BARKER_Tu * 1e6:.2f} мкс",
                "Количество импульсов": "1"
            },
            resolutions={
                "Разрешение по дальности (ΔR)": f"{delta_R_barker:.1f} м",
                "Разрешение по скорости (ΔV)": f"{delta_V_barker:.2f} м/с",
                "Уровень боковых лепестков АКФ": f"1/{len(BARKER_CODE)} = {side_lobe_level:.3f} ({side_lobe_db:.1f} дБ)",
                "База сигнала": str(len(BARKER_CODE))
            }
        )
        
        # Задание 4: М-последовательность
        mseq_str = ''.join(['1' if x == 1 else '-1' for x in MSEQ_CODE])
        delta_R_mseq = c0 * MSEQ_Tu / 2
        delta_V_mseq = (1/(len(MSEQ_CODE)*MSEQ_Tu)) * lambda_c / 2
        
        self.create_info_block(
            task_num=4,
            title="Задание 4: М-последовательность",
            signal_type="Псевдослучайная М-последовательность",
            params={
                "Длина последовательности": str(len(MSEQ_CODE)),
                "Кодовая последовательность": mseq_str,
                "Длительность элементарного импульса": f"{MSEQ_Tu*1e6:.2f} мкс",
                "Общая длительность сигнала": f"{len(MSEQ_CODE) * MSEQ_Tu * 1e6:.2f} мкс",
                "Количество импульсов": "1"
            },
            resolutions={
                "Разрешение по дальности (ΔR)": f"{delta_R_mseq:.1f} м",
                "Разрешение по скорости (ΔV)": f"{delta_V_mseq:.2f} м/с",
                "Уровень боковых лепестков АКФ": f"~1/{len(MSEQ_CODE)} (для M-последовательности)",
                "База сигнала": str(len(MSEQ_CODE))
            }
        )
        
        # Задание 5: Пачка прямоугольных импульсов
        delta_R_burst = c0 * BURST_Tu / 2
        delta_V_burst = c0 / (2 * f0 * BURST_N_pt * BURST_Tpr)
        R_unambiguous = c0 * BURST_Tpr / 2
        V_unambiguous = c0 / (2 * f0 * BURST_Tpr)
        
        self.create_info_block(
            task_num=5,
            title="Задание 5: Пачка прямоугольных импульсов",
            signal_type="Пачка прямоугольных импульсов",
            params={
                "Длительность импульса (τи)": f"{BURST_Tu*1e6:.2f} мкс",
                "Период повторения (Tпр)": f"{BURST_Tpr*1e6:.2f} мкс",
                "Количество импульсов в пачке": str(BURST_N_pt),
                "Длительность пачки": f"{BURST_N_pt * BURST_Tpr * 1e6:.1f} мкс"
            },
            resolutions={
                "Разрешение по дальности (ΔR)": f"{delta_R_burst:.1f} м",
                "Разрешение по скорости (ΔV)": f"{delta_V_burst:.2f} м/с",
                "Период неоднозначности по дальности": f"{R_unambiguous:.1f} м",
                "Период неоднозначности по скорости": f"{V_unambiguous:.1f} м/с"
            }
        )
        
        # Задание 6: R/V диаграмма для пачки прямоугольных импульсов
        self.create_info_block(
            task_num=6,
            title="Задание 6: R/V диаграмма",
            signal_type="Пачка прямоугольных импульсов",
            params={
                "Длительность импульса (τи)": f"{BURST_Tu*1e6:.2f} мкс",
                "Период повторения (Tпр)": f"{BURST_Tpr*1e6:.2f} мкс",
                "Количество импульсов": str(BURST_N_pt),
                "Отображение": "Функция неопределенности в координатах (дальность, скорость)"
            },
            resolutions={
                "Разрешение по дальности (ΔR)": f"{delta_R_burst:.1f} м",
                "Разрешение по скорости (ΔV)": f"{delta_V_burst:.2f} м/с",
                "Максимальная однозначная дальность": f"{R_unambiguous:.1f} м",
                "Максимальная однозначная скорость": f"{V_unambiguous:.1f} м/с"
            }
        )
        
        # Задание 7: Пачка ЛЧМ сигналов
        delta_R_lfm_burst = c0 / (2 * LFM_BURST_deltaF)
        delta_V_lfm_burst = c0 / (2 * f0 * LFM_BURST_N_pt * LFM_BURST_Tpr)
        R_unambiguous_lfm_burst = c0 * LFM_BURST_Tpr / 2
        V_unambiguous_lfm_burst = c0 / (2 * f0 * LFM_BURST_Tpr)
        base_lfm_burst = LFM_BURST_Tu * LFM_BURST_deltaF
        
        self.create_info_block(
            task_num=7,
            title="Задание 7: Пачка ЛЧМ сигналов",
            signal_type="Пачка линейно-частотно-модулированных импульсов",
            params={
                "Длительность импульса (τи)": f"{LFM_BURST_Tu*1e6:.2f} мкс",
                "Период повторения (Tпр)": f"{LFM_BURST_Tpr*1e6:.2f} мкс",
                "Девиация частоты (Δf)": f"{LFM_BURST_deltaF/1e6:.0f} МГц",
                "Количество импульсов в пачке": str(LFM_BURST_N_pt),
                "Длительность пачки": f"{LFM_BURST_N_pt * LFM_BURST_Tpr * 1e6:.1f} мкс",
                "База одного импульса": f"{base_lfm_burst:.0f}"
            },
            resolutions={
                "Разрешение по дальности (после сжатия)": f"{delta_R_lfm_burst:.1f} м",
                "Разрешение по скорости (пачка)": f"{delta_V_lfm_burst:.2f} м/с",
                "Период неоднозначности по дальности": f"{R_unambiguous_lfm_burst:.1f} м",
                "Период неоднозначности по скорости": f"{V_unambiguous_lfm_burst:.1f} м/с",
                "Коэффициент сжатия ЛЧМ": f"{base_lfm_burst:.0f}"
            }
        )
        
    def create_info_block(self, task_num, title, signal_type, params, resolutions):
        """Создание информационного блока для задания"""
        # Основной фрейм
        frame = ctk.CTkFrame(self.info_scrollable, corner_radius=8)
        frame.pack(fill="x", padx=5, pady=8)
        self.info_frames[task_num] = frame
        
        # Цветовая схема для разных заданий
        colors = ["#1e3a5f", "#2e5a7f", "#3e6a8f", "#4e7a9f", "#5e8aaf", "#6e9abf", "#7eaacf"]
        color = colors[task_num - 1] if task_num <= len(colors) else "#8ebadf"
        
        # Заголовок
        title_frame = ctk.CTkFrame(frame, fg_color=color, corner_radius=8)
        title_frame.pack(fill="x", padx=2, pady=2)
        
        title_label = ctk.CTkLabel(title_frame, text=title, 
                                    font=ctk.CTkFont(size=13, weight="bold"))
        title_label.pack(pady=5)
        
        # Тип сигнала
        type_label = ctk.CTkLabel(title_frame, text=signal_type, 
                                   font=ctk.CTkFont(size=11, slant="italic"))
        type_label.pack(pady=2)
        
        # Параметры сигнала
        params_frame = ctk.CTkFrame(frame, corner_radius=8)
        params_frame.pack(fill="x", padx=5, pady=5)
        
        params_title = ctk.CTkLabel(params_frame, text="Параметры сигнала:", 
                                     font=ctk.CTkFont(size=12, weight="bold"))
        params_title.pack(pady=3, anchor="w", padx=5)
        
        for key, value in params.items():
            if "последовательность" in key:
                param_label = ctk.CTkLabel(params_frame, text=f"  {key}: {value}", 
                                            font=ctk.CTkFont(size=10, family="Consolas"))
            else:
                param_label = ctk.CTkLabel(params_frame, text=f"  {key}: {value}", 
                                            font=ctk.CTkFont(size=11))
            param_label.pack(pady=1, anchor="w", padx=15)
        
        # Разделитель
        separator = ctk.CTkFrame(frame, height=2, fg_color="gray")
        separator.pack(fill="x", padx=5, pady=3)
        
        # Разрешающие способности
        res_frame = ctk.CTkFrame(frame, corner_radius=8)
        res_frame.pack(fill="x", padx=5, pady=5)
        
        res_title = ctk.CTkLabel(res_frame, text="Разрешающие способности:", 
                                  font=ctk.CTkFont(size=12, weight="bold"))
        res_title.pack(pady=3, anchor="w", padx=5)
        
        for key, value in resolutions.items():
            res_label = ctk.CTkLabel(res_frame, text=f"  {key}: {value}", 
                                      font=ctk.CTkFont(size=11))
            res_label.pack(pady=1, anchor="w", padx=15)
            
    def add_general_info(self):
        """Добавление общей информации"""
        general_frame = ctk.CTkFrame(self.info_scrollable, corner_radius=8)
        general_frame.pack(fill="x", padx=5, pady=10)
        
        general_title = ctk.CTkLabel(general_frame, text="Общие параметры", 
                                      font=ctk.CTkFont(size=14, weight="bold"))
        general_title.pack(pady=5)
        
        general_params = [
            f"Скорость света (c0): {c0/1e8:.1f}x10⁸ м/с",
            f"Рабочая частота (f0): {f0/1e9:.0f} ГГц",
            f"Длина волны (λ): {lambda_c*100:.0f} см",
            f"Промежуточная частота (IF): {IF/1e6:.0f} МГц",
            f"Частота дискретизации (Fs): {Fs/1e6:.0f} МГц",
            f"Время анализа (Ta): {Ta*1e6:.1f} мкс"
        ]
        
        for param in general_params:
            label = ctk.CTkLabel(general_frame, text=param, 
                                 font=ctk.CTkFont(size=11), justify="left")
            label.pack(pady=2, padx=10, anchor="w")
            
    def create_figure(self, title, figsize=(12, 8)):
        """Создание новой фигуры для графика"""
        fig = Figure(figsize=figsize, facecolor='#2b2b2b')
        fig.suptitle(title, fontsize=14, color='white', fontweight='bold')
        return fig
    
    def add_plot_to_frame(self, fig, signal_name, task_num):
        """Добавление графика в прокручиваемую область с привязкой к заданию"""
        # Создаем фрейм для графика
        plot_frame = ctk.CTkFrame(self.plots_frame, corner_radius=10)
        plot_frame.pack(fill="x", expand=False, pady=15, padx=10)
        
        # Заголовок графика
        title_label = ctk.CTkLabel(plot_frame, text=signal_name, 
                                    font=ctk.CTkFont(size=16, weight="bold"))
        title_label.pack(pady=10)
        
        # Создаем canvas для matplotlib
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        
        # Добавляем тулбар для интерактивного управления
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        toolbar = NavigationToolbar2Tk(canvas, plot_frame, pack_toolbar=True)
        toolbar.update()
        
        # Упаковываем canvas
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
    def plot_3d_ambiguity(self, U, signal_name, tau_limits, f_limits, N_tau=300):
        """Построение 3D функции неопределенности"""
        tau_min, tau_max = tau_limits
        f_max_display = f_limits
        
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, N_tau, f_max_display)
        
        # Создание 3D графика
        fig = Figure(figsize=(12, 8), facecolor='#2b2b2b')
        fig.suptitle(f'3D Функция неопределенности - {signal_name}', fontsize=14, color='white', fontweight='bold')
        
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#2b2b2b')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.zaxis.label.set_color('white')
        ax.tick_params(colors='white')
        
        # Создание сетки
        X, Y = np.meshgrid(f_plot, tau_af*1e6)
        
        # Построение поверхности
        surf = ax.plot_surface(X, Y, amf, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
        
        # Настройка осей
        ax.set_xlabel('f - f0, МГц', fontsize=11, labelpad=10)
        ax.set_ylabel('τ, мкс', fontsize=11, labelpad=10)
        ax.set_zlabel('|χ(τ,f)|, отн.ед.', fontsize=11, labelpad=10)
        
        # Добавление colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20)
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        cbar.set_label('Нормированная амплитуда', color='white')
        
        # Настройка угла обзора
        ax.view_init(elev=25, azim=-60)
        
        fig.tight_layout()
        return fig
        
    def plot_pulse_signal(self):
        """Построение графиков для прямоугольного импульса"""
        U = generate_pulse_signal()
        
        S = fft(U)
        B = correlate(U, U, mode='full')
        tau = np.linspace(-Ta, Ta, len(B))
        
        tau_min, tau_max = -PULSE_Tu, PULSE_Tu
        f_max_display = 40e6
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, 200, f_max_display)
        
        fig = self.create_figure(f'Прямоугольный импульс (τи = {PULSE_Tu*1e6:.2f} мкс)')
        axes = fig.subplots(2, 2)
        
        for ax in axes.flat:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        axes[0, 0].plot(t*1e6, np.real(U), 'b', linewidth=0.8)
        axes[0, 0].set_xlabel('t, мкс')
        axes[0, 0].set_ylabel('U(t), B')
        axes[0, 0].set_title('Исходный сигнал')
        axes[0, 0].grid(True, alpha=0.3)
        
        f_plot_fft = f - IF
        idx_plot = np.abs(f_plot_fft) <= 40e6
        axes[0, 1].plot(f_plot_fft[idx_plot]/1e6, np.abs(S[idx_plot]), 'g', linewidth=0.8)
        axes[0, 1].set_xlabel('f - f0, МГц')
        axes[0, 1].set_ylabel('|S(f)|')
        axes[0, 1].set_title('Амплитудный спектр')
        axes[0, 1].grid(True, alpha=0.3)
        
        B_norm = np.abs(B) / np.max(np.abs(B))
        tau_us = tau * 1e6
        idx_akf = np.abs(tau_us) <= 2
        axes[1, 0].plot(tau_us[idx_akf], B_norm[idx_akf], 'm', linewidth=0.8)
        axes[1, 0].set_xlabel('τ, мкс')
        axes[1, 0].set_ylabel('|B(τ)|, отн.ед.')
        axes[1, 0].set_title('Нормированная АКФ')
        axes[1, 0].grid(True, alpha=0.3)
        
        contour_levels = [0.1, 0.5, 0.707]
        contour = axes[1, 1].contour(f_plot, tau_af*1e6, amf, contour_levels, colors='white', linewidths=1.5)
        axes[1, 1].clabel(contour, inline=True, fontsize=8, colors='white')
        axes[1, 1].set_xlabel('f - f0, МГц')
        axes[1, 1].set_ylabel('τ, мкс')
        axes[1, 1].set_title('Сечения функции неопределенности')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        fig.tight_layout()
        self.add_plot_to_frame(fig, "1. Прямоугольный импульс", 1)
        
        # 3D график
        fig_3d = self.plot_3d_ambiguity(U, "Прямоугольный импульс", (-PULSE_Tu, PULSE_Tu), 40e6)
        self.add_plot_to_frame(fig_3d, "1.3D Прямоугольный импульс (ФН)", 1)
        
    def plot_lfm_signal(self):
        """Построение графиков для сигнала с ЛЧМ"""
        U = generate_lfm_signal()
        
        S = fft(U)
        B = correlate(U, U, mode='full')
        tau = np.linspace(-Ta, Ta, len(B))
        
        tau_min, tau_max = -LFM_Tu, LFM_Tu
        f_max_display = 25e6
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, 200, f_max_display)
        
        fig = self.create_figure(f'Сигнал с ЛЧМ (τи = {LFM_Tu*1e6:.2f} мкс, Δf = {LFM_deltaF/1e6:.0f} МГц)')
        axes = fig.subplots(2, 2)
        
        for ax in axes.flat:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        axes[0, 0].plot(t*1e6, np.real(U), 'b', linewidth=0.8)
        axes[0, 0].set_xlabel('t, мкс')
        axes[0, 0].set_ylabel('U(t), B')
        axes[0, 0].set_title('Исходный сигнал (Re)')
        axes[0, 0].grid(True, alpha=0.3)
        
        f_plot_fft = f - IF
        idx_plot = np.abs(f_plot_fft) <= 25e6
        axes[0, 1].plot(f_plot_fft[idx_plot]/1e6, np.abs(S[idx_plot]), 'g', linewidth=0.8)
        axes[0, 1].set_xlabel('f - f0, МГц')
        axes[0, 1].set_ylabel('|S(f)|')
        axes[0, 1].set_title('Амплитудный спектр')
        axes[0, 1].grid(True, alpha=0.3)
        
        B_norm = np.abs(B) / np.max(np.abs(B))
        tau_us = tau * 1e6
        idx_akf = np.abs(tau_us) <= 2
        axes[1, 0].plot(tau_us[idx_akf], B_norm[idx_akf], 'm', linewidth=0.8)
        axes[1, 0].set_xlabel('τ, мкс')
        axes[1, 0].set_ylabel('|B(τ)|, отн.ед.')
        axes[1, 0].set_title('Нормированная АКФ')
        axes[1, 0].grid(True, alpha=0.3)
        
        X, Y = np.meshgrid(f_plot, tau_af*1e6)
        surf = axes[1, 1].pcolormesh(X, Y, 20*np.log10(amf + 1e-10), shading='auto', cmap='jet_r')
        axes[1, 1].set_xlabel('f - f0, МГц')
        axes[1, 1].set_ylabel('τ, мкс')
        axes[1, 1].set_title('Функция неопределенности (дБ)')
        plt.colorbar(surf, ax=axes[1, 1], label='дБ')
        
        fig.tight_layout()
        self.add_plot_to_frame(fig, "2. Сигнал с ЛЧМ", 2)
        
        # 3D график
        fig_3d = self.plot_3d_ambiguity(U, "Сигнал с ЛЧМ", (-LFM_Tu, LFM_Tu), 25e6)
        self.add_plot_to_frame(fig_3d, "2.3D Сигнал с ЛЧМ (ФН)", 2)
        
    def plot_barker_signal(self):
        """Построение графиков для кода Баркера"""
        U = generate_barker_signal()
        
        S = fft(U)
        B = correlate(U, U, mode='full')
        tau = np.linspace(-Ta, Ta, len(B))
        
        tau_min, tau_max = -10*BARKER_Tu, 10*BARKER_Tu
        f_max_display = 8e6
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, 300, f_max_display)
        
        # Восстановление огибающей для отображения
        S_ampl = np.zeros(Nt, dtype=float)
        for i, val in enumerate(BARKER_CODE):
            mask = (t >= i * BARKER_Tu) & (t < i * BARKER_Tu + BARKER_Tu)
            S_ampl[mask] = val
        
        fig = self.create_figure(f'Код Баркера (длина {len(BARKER_CODE)}, τэ = {BARKER_Tu*1e6:.2f} мкс)')
        axes = fig.subplots(2, 2)
        
        for ax in axes.flat:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        axes[0, 0].plot(t*1e6, S_ampl, 'b', linewidth=0.8)
        axes[0, 0].set_xlabel('t, мкс')
        axes[0, 0].set_ylabel('A(t), отн.ед.')
        axes[0, 0].set_title('Огибающая сигнала')
        axes[0, 0].grid(True, alpha=0.3)
        
        f_plot_fft = f - IF
        idx_plot = np.abs(f_plot_fft) <= 8e6
        axes[0, 1].plot(f_plot_fft[idx_plot]/1e6, np.abs(S[idx_plot]), 'g', linewidth=0.8)
        axes[0, 1].set_xlabel('f - f0, МГц')
        axes[0, 1].set_ylabel('|S(f)|')
        axes[0, 1].set_title('Амплитудный спектр')
        axes[0, 1].grid(True, alpha=0.3)
        
        B_norm = np.abs(B) / np.max(np.abs(B))
        tau_us = tau * 1e6
        idx_akf = np.abs(tau_us) <= 4
        axes[1, 0].plot(tau_us[idx_akf], B_norm[idx_akf], 'm', linewidth=0.8)
        axes[1, 0].set_xlabel('τ, мкс')
        axes[1, 0].set_ylabel('|B(τ)|, отн.ед.')
        axes[1, 0].set_title('Нормированная АКФ')
        axes[1, 0].grid(True, alpha=0.3)
        
        contour_levels = [0.1, 0.5, 0.707]
        contour = axes[1, 1].contour(f_plot, tau_af*1e6, amf, contour_levels, colors='white', linewidths=1.5)
        axes[1, 1].clabel(contour, inline=True, fontsize=8, colors='white')
        axes[1, 1].set_xlabel('f - f0, МГц')
        axes[1, 1].set_ylabel('τ, мкс')
        axes[1, 1].set_title('Сечения функции неопределенности')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        self.add_plot_to_frame(fig, "3. Код Баркера", 3)
        
        # 3D график
        fig_3d = self.plot_3d_ambiguity(U, "Код Баркера", (-10*BARKER_Tu, 10*BARKER_Tu), 8e6)
        self.add_plot_to_frame(fig_3d, "3.3D Код Баркера (ФН)", 3)
        
    def plot_msequence_signal(self):
        """Построение графиков для М-последовательности"""
        U = generate_msequence_signal()
        
        S = fft(U)
        B = correlate(U, U, mode='full')
        tau = np.linspace(-Ta, Ta, len(B))
        
        tau_min, tau_max = -10*MSEQ_Tu, 10*MSEQ_Tu
        f_max_display = 8e6
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, 300, f_max_display)
        
        # Восстановление огибающей для отображения
        S_ampl = np.zeros(Nt, dtype=float)
        for i, val in enumerate(MSEQ_CODE):
            mask = (t >= i * MSEQ_Tu) & (t < i * MSEQ_Tu + MSEQ_Tu)
            S_ampl[mask] = val
        
        fig = self.create_figure(f'М-последовательность (длина {len(MSEQ_CODE)}, τэ = {MSEQ_Tu*1e6:.2f} мкс)')
        axes = fig.subplots(2, 2)
        
        for ax in axes.flat:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        axes[0, 0].plot(t*1e6, S_ampl, 'b', linewidth=0.8)
        axes[0, 0].set_xlabel('t, мкс')
        axes[0, 0].set_ylabel('A(t), отн.ед.')
        axes[0, 0].set_title('Огибающая сигнала')
        axes[0, 0].grid(True, alpha=0.3)
        
        f_plot_fft = f - IF
        idx_plot = np.abs(f_plot_fft) <= 8e6
        axes[0, 1].plot(f_plot_fft[idx_plot]/1e6, np.abs(S[idx_plot]), 'g', linewidth=0.8)
        axes[0, 1].set_xlabel('f - f0, МГц')
        axes[0, 1].set_ylabel('|S(f)|')
        axes[0, 1].set_title('Амплитудный спектр')
        axes[0, 1].grid(True, alpha=0.3)
        
        B_norm = np.abs(B) / np.max(np.abs(B))
        tau_us = tau * 1e6
        idx_akf = np.abs(tau_us) <= 4
        axes[1, 0].plot(tau_us[idx_akf], B_norm[idx_akf], 'm', linewidth=0.8)
        axes[1, 0].set_xlabel('τ, мкс')
        axes[1, 0].set_ylabel('|B(τ)|, отн.ед.')
        axes[1, 0].set_title('Нормированная АКФ')
        axes[1, 0].grid(True, alpha=0.3)
        
        contour_levels = [0.1, 0.5, 0.707]
        contour = axes[1, 1].contour(f_plot, tau_af*1e6, amf, contour_levels, colors='white', linewidths=1.5)
        axes[1, 1].clabel(contour, inline=True, fontsize=8, colors='white')
        axes[1, 1].set_xlabel('f - f0, МГц')
        axes[1, 1].set_ylabel('τ, мкс')
        axes[1, 1].set_title('Сечения функции неопределенности')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        self.add_plot_to_frame(fig, "4. М-последовательность", 4)
        
        # 3D график
        fig_3d = self.plot_3d_ambiguity(U, "М-последовательность", (-10*MSEQ_Tu, 10*MSEQ_Tu), 8e6)
        self.add_plot_to_frame(fig_3d, "4.3D М-последовательность (ФН)", 4)
        
    def plot_burst_signal(self):
        """Построение графиков для пачки прямоугольных импульсов"""
        U = generate_burst_signal()
        
        S = fft(U)
        B = correlate(U, U, mode='full')
        tau = np.linspace(-Ta, Ta, len(B))
        
        tau_min, tau_max = -8*BURST_Tu, 8*BURST_Tu
        f_max_display = 40e6
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, 300, f_max_display)
        
        # Восстановление огибающей для отображения
        S_ampl = np.zeros(Nt, dtype=float)
        for n in range(BURST_N_pt):
            mask = (t >= n * BURST_Tpr) & (t < n * BURST_Tpr + BURST_Tu)
            S_ampl[mask] = 1.0
        
        fig = self.create_figure(f'Пачка прямоугольных импульсов (N = {BURST_N_pt}, Tпр = {BURST_Tpr*1e6:.2f} мкс)')
        axes = fig.subplots(2, 2)
        
        for ax in axes.flat:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        axes[0, 0].plot(t*1e6, S_ampl, 'b', linewidth=0.8)
        axes[0, 0].set_xlabel('t, мкс')
        axes[0, 0].set_ylabel('A(t), отн.ед.')
        axes[0, 0].set_title('Огибающая пачки')
        axes[0, 0].grid(True, alpha=0.3)
        
        f_plot_fft = f - IF
        idx_plot = np.abs(f_plot_fft) <= 40e6
        axes[0, 1].plot(f_plot_fft[idx_plot]/1e6, np.abs(S[idx_plot]), 'g', linewidth=0.8)
        axes[0, 1].set_xlabel('f - f0, МГц')
        axes[0, 1].set_ylabel('|S(f)|')
        axes[0, 1].set_title('Амплитудный спектр')
        axes[0, 1].grid(True, alpha=0.3)
        
        B_norm = np.abs(B) / np.max(np.abs(B))
        tau_us = tau * 1e6
        idx_akf = np.abs(tau_us) <= 10
        axes[1, 0].plot(tau_us[idx_akf], B_norm[idx_akf], 'm', linewidth=0.8)
        axes[1, 0].set_xlabel('τ, мкс')
        axes[1, 0].set_ylabel('|B(τ)|, отн.ед.')
        axes[1, 0].set_title('Нормированная АКФ')
        axes[1, 0].grid(True, alpha=0.3)
        
        contour_levels = [0.1, 0.5, 0.707]
        contour = axes[1, 1].contour(f_plot, tau_af*1e6, amf, contour_levels, colors='white', linewidths=1.5)
        axes[1, 1].clabel(contour, inline=True, fontsize=8, colors='white')
        axes[1, 1].set_xlabel('f - f0, МГц')
        axes[1, 1].set_ylabel('τ, мкс')
        axes[1, 1].set_title('Сечения функции неопределенности')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        fig.tight_layout()
        self.add_plot_to_frame(fig, "5. Пачка прямоугольных импульсов", 5)
        
        # 3D график
        fig_3d = self.plot_3d_ambiguity(U, "Пачка прямоугольных импульсов", (-8*BURST_Tu, 8*BURST_Tu), 40e6)
        self.add_plot_to_frame(fig_3d, "5.3D Пачка прямоугольных импульсов (ФН)", 5)
        
    def plot_rv_diagram(self):
        """Построение R/V диаграммы для пачки прямоугольных импульсов"""
        U = generate_burst_signal()
        
        tau_min, tau_max = -8*BURST_Tu, 8*BURST_Tu
        f_max_display = 40e6
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, 300, f_max_display)
        
        # Пересчет в координаты дальности и скорости
        R_tau = tau_af * c0 / 2
        V_f = f_plot * 1e6 * c0 / (2 * f0)
        
        # Физические пределы
        R_max_unambiguous = c0 * BURST_Tpr / 2
        delta_R = c0 * BURST_Tu / 2
        delta_V = c0 / (2 * f0 * BURST_N_pt * BURST_Tpr)
        
        R_limit = min(3 * R_max_unambiguous, 1000)
        V_limit = 3 * c0 / (2 * f0 * BURST_Tpr)
        
        mask_R = np.abs(R_tau) <= R_limit
        mask_V = np.abs(V_f) <= V_limit
        R_tau_cut = R_tau[mask_R]
        V_f_cut = V_f[mask_V]
        amf_cut = amf[np.ix_(mask_R, mask_V)]
        
        # Создание фигуры
        fig = Figure(figsize=(12, 10), facecolor='#2b2b2b')
        fig.suptitle('Функция неопределенности пачки прямоугольных импульсов в координатах (дальность, скорость)', 
                     fontsize=14, color='white', fontweight='bold')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        
        X_grid, Y_grid = np.meshgrid(V_f_cut, R_tau_cut)
        levels = np.linspace(0, 1, 50)
        cf = ax.contourf(X_grid, Y_grid, amf_cut, levels=levels, cmap='viridis', alpha=0.9)
        
        contour_levels_rv = [0.1, 0.5, 0.707]
        contour_RV = ax.contour(X_grid, Y_grid, amf_cut, contour_levels_rv, 
                                 colors='red', linewidths=2.5, linestyles='solid')
        ax.clabel(contour_RV, inline=True, fontsize=11, fmt='%.1f', colors='white')
        
        cbar = plt.colorbar(cf, ax=ax, label='|χ(τ,f)|, отн.ед.')
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        
        ax.set_xlabel(f'Скорость V, м/с (ΔV = {delta_V:.1f} м/с)', fontsize=12)
        ax.set_ylabel(f'Дальность R, м (ΔR = {delta_R:.1f} м, R_max = {R_max_unambiguous:.0f} м)', fontsize=12)
        ax.set_title('Функция неопределенности в координатах (дальность, скорость)', fontsize=13)
        
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.set_xlim([-V_limit, V_limit])
        ax.set_ylim([-R_limit, R_limit])
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.text(0, 0, 'Главный лепесток', ha='center', va='center', 
                fontsize=10, color='white', fontweight='bold')
        
        fig.tight_layout()
        self.add_plot_to_frame(fig, "6. R/V диаграмма для пачки прямоугольных импульсов", 6)
        
        # 3D график для R/V диаграммы
        fig_3d = Figure(figsize=(12, 8), facecolor='#2b2b2b')
        fig_3d.suptitle('3D Функция неопределенности в координатах (дальность, скорость)', 
                        fontsize=14, color='white', fontweight='bold')
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.set_facecolor('#2b2b2b')
        ax_3d.xaxis.label.set_color('white')
        ax_3d.yaxis.label.set_color('white')
        ax_3d.zaxis.label.set_color('white')
        ax_3d.tick_params(colors='white')
        
        X_grid_3d, Y_grid_3d = np.meshgrid(V_f_cut, R_tau_cut)
        surf = ax_3d.plot_surface(X_grid_3d, Y_grid_3d, amf_cut, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
        
        ax_3d.set_xlabel('Скорость V, м/с', fontsize=11, labelpad=10)
        ax_3d.set_ylabel('Дальность R, м', fontsize=11, labelpad=10)
        ax_3d.set_zlabel('|χ(τ,f)|, отн.ед.', fontsize=11, labelpad=10)
        
        cbar = fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=20)
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        cbar.set_label('Нормированная амплитуда', color='white')
        
        ax_3d.view_init(elev=25, azim=-60)
        
        fig_3d.tight_layout()
        self.add_plot_to_frame(fig_3d, "6.3D R/V диаграмма (ФН)", 6)
        
    def plot_lfm_burst_signal(self):
        """Построение графиков для пачки ЛЧМ сигналов"""
        U = generate_lfm_burst_signal()
        
        S = fft(U)
        B = correlate(U, U, mode='full')
        tau = np.linspace(-Ta, Ta, len(B))
        
        tau_min, tau_max = -8*LFM_BURST_Tu, 8*LFM_BURST_Tu
        f_max_display = 15e6
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, 300, f_max_display)
        
        # Восстановление огибающей для отображения
        S_ampl = np.zeros(Nt, dtype=float)
        for n in range(LFM_BURST_N_pt):
            mask = (t >= n * LFM_BURST_Tpr) & (t < n * LFM_BURST_Tpr + LFM_BURST_Tu)
            S_ampl[mask] = 1.0
        
        fig = self.create_figure(f'Пачка ЛЧМ сигналов (N = {LFM_BURST_N_pt}, Tпр = {LFM_BURST_Tpr*1e6:.2f} мкс, Δf = {LFM_BURST_deltaF/1e6:.0f} МГц)')
        axes = fig.subplots(2, 2)
        
        for ax in axes.flat:
            ax.set_facecolor('#2b2b2b')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
        
        # Огибающая пачки (действительная часть для ЛЧМ)
        axes[0, 0].plot(t*1e6, np.real(U), 'b', linewidth=0.8)
        axes[0, 0].set_xlabel('t, мкс')
        axes[0, 0].set_ylabel('U(t), B')
        axes[0, 0].set_title('Пачка ЛЧМ сигналов (Re)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Спектр
        f_plot_fft = f - IF
        idx_plot = np.abs(f_plot_fft) <= 15e6
        axes[0, 1].plot(f_plot_fft[idx_plot]/1e6, np.abs(S[idx_plot]), 'g', linewidth=0.8)
        axes[0, 1].set_xlabel('f - f0, МГц')
        axes[0, 1].set_ylabel('|S(f)|')
        axes[0, 1].set_title('Амплитудный спектр')
        axes[0, 1].grid(True, alpha=0.3)
        
        # АКФ
        B_norm = np.abs(B) / np.max(np.abs(B))
        tau_us = tau * 1e6
        idx_akf = np.abs(tau_us) <= 12
        axes[1, 0].plot(tau_us[idx_akf], B_norm[idx_akf], 'm', linewidth=0.8)
        axes[1, 0].set_xlabel('τ, мкс')
        axes[1, 0].set_ylabel('|B(τ)|, отн.ед.')
        axes[1, 0].set_title('Нормированная АКФ')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Функция неопределенности (2D сечение)
        X, Y = np.meshgrid(f_plot, tau_af*1e6)
        surf = axes[1, 1].pcolormesh(X, Y, 20*np.log10(amf + 1e-10), shading='auto', cmap='jet_r')
        axes[1, 1].set_xlabel('f - f0, МГц')
        axes[1, 1].set_ylabel('τ, мкс')
        axes[1, 1].set_title('Функция неопределенности (дБ)')
        plt.colorbar(surf, ax=axes[1, 1], label='дБ')
        
        fig.tight_layout()
        self.add_plot_to_frame(fig, "7. Пачка ЛЧМ сигналов", 7)
        
        # 3D график
        fig_3d = self.plot_3d_ambiguity(U, "Пачка ЛЧМ сигналов", (-8*LFM_BURST_Tu, 8*LFM_BURST_Tu), 15e6)
        self.add_plot_to_frame(fig_3d, "7.3D Пачка ЛЧМ сигналов (ФН)", 7)
        
        # R/V диаграмма для пачки ЛЧМ
        self.plot_lfm_burst_rv_diagram(U)
        
    def plot_lfm_burst_rv_diagram(self, U):
        """Построение R/V диаграммы для пачки ЛЧМ сигналов"""
        tau_min, tau_max = -8*LFM_BURST_Tu, 8*LFM_BURST_Tu
        f_max_display = 15e6
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, 300, f_max_display)
        
        # Пересчет в координаты дальности и скорости
        R_tau = tau_af * c0 / 2
        V_f = f_plot * 1e6 * c0 / (2 * f0)
        
        # Физические пределы
        R_max_unambiguous = c0 * LFM_BURST_Tpr / 2
        delta_R = c0 / (2 * LFM_BURST_deltaF)
        delta_V = c0 / (2 * f0 * LFM_BURST_N_pt * LFM_BURST_Tpr)
        
        R_limit = min(3 * R_max_unambiguous, 1500)
        V_limit = 3 * c0 / (2 * f0 * LFM_BURST_Tpr)
        
        mask_R = np.abs(R_tau) <= R_limit
        mask_V = np.abs(V_f) <= V_limit
        R_tau_cut = R_tau[mask_R]
        V_f_cut = V_f[mask_V]
        amf_cut = amf[np.ix_(mask_R, mask_V)]
        
        # Создание фигуры
        fig = Figure(figsize=(12, 10), facecolor='#2b2b2b')
        fig.suptitle('Функция неопределенности пачки ЛЧМ сигналов в координатах (дальность, скорость)', 
                     fontsize=14, color='white', fontweight='bold')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        
        X_grid, Y_grid = np.meshgrid(V_f_cut, R_tau_cut)
        levels = np.linspace(0, 1, 50)
        cf = ax.contourf(X_grid, Y_grid, amf_cut, levels=levels, cmap='viridis', alpha=0.9)
        
        contour_levels_rv = [0.1, 0.5, 0.707]
        contour_RV = ax.contour(X_grid, Y_grid, amf_cut, contour_levels_rv, 
                                 colors='red', linewidths=2.5, linestyles='solid')
        ax.clabel(contour_RV, inline=True, fontsize=11, fmt='%.1f', colors='white')
        
        cbar = plt.colorbar(cf, ax=ax, label='|χ(τ,f)|, отн.ед.')
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        
        ax.set_xlabel(f'Скорость V, м/с (ΔV = {delta_V:.1f} м/с)', fontsize=12)
        ax.set_ylabel(f'Дальность R, м (ΔR = {delta_R:.1f} м, R_max = {R_max_unambiguous:.0f} м)', fontsize=12)
        ax.set_title('Функция неопределенности в координатах (дальность, скорость) для пачки ЛЧМ', fontsize=13)
        
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.set_xlim([-V_limit, V_limit])
        ax.set_ylim([-R_limit, R_limit])
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.text(0, 0, 'Главный\nлепесток', ha='center', va='center', 
                fontsize=10, color='white', fontweight='bold')
        
        fig.tight_layout()
        self.add_plot_to_frame(fig, "7. R/V диаграмма для пачки ЛЧМ сигналов", 7)
        
        # 3D график для R/V диаграммы
        fig_3d = Figure(figsize=(12, 8), facecolor='#2b2b2b')
        fig_3d.suptitle('3D Функция неопределенности пачки ЛЧМ в координатах (дальность, скорость)', 
                        fontsize=14, color='white', fontweight='bold')
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        ax_3d.set_facecolor('#2b2b2b')
        ax_3d.xaxis.label.set_color('white')
        ax_3d.yaxis.label.set_color('white')
        ax_3d.zaxis.label.set_color('white')
        ax_3d.tick_params(colors='white')
        
        X_grid_3d, Y_grid_3d = np.meshgrid(V_f_cut, R_tau_cut)
        surf = ax_3d.plot_surface(X_grid_3d, Y_grid_3d, amf_cut, cmap='viridis', alpha=0.9, linewidth=0, antialiased=True)
        
        ax_3d.set_xlabel('Скорость V, м/с', fontsize=11, labelpad=10)
        ax_3d.set_ylabel('Дальность R, м', fontsize=11, labelpad=10)
        ax_3d.set_zlabel('|χ(τ,f)|, отн.ед.', fontsize=11, labelpad=10)
        
        cbar = fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=20)
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        cbar.set_label('Нормированная амплитуда', color='white')
        
        ax_3d.view_init(elev=25, azim=-60)
        
        fig_3d.tight_layout()
        self.add_plot_to_frame(fig_3d, "7.3D R/V диаграмма для пачки ЛЧМ (ФН)", 7)
        
    def generate_all_plots(self):
        """Генерация и отображение всех графиков"""
        print("\n" + "="*70)
        print(" ГЕНЕРАЦИЯ ВСЕХ ГРАФИКОВ")
        print("="*70)
        print("\nВыполняется расчет и построение графиков...\n")
        
        self.plot_pulse_signal()
        print(" 1. Прямоугольный импульс + 3D ФН")
        
        self.plot_lfm_signal()
        print(" 2. Сигнал с ЛЧМ + 3D ФН")
        
        self.plot_barker_signal()
        print(" 3. Код Баркера + 3D ФН")
        
        self.plot_msequence_signal()
        print(" 4. М-последовательность + 3D ФН")
        
        self.plot_burst_signal()
        print(" 5. Пачка прямоугольных импульсов + 3D ФН")
        
        self.plot_rv_diagram()
        print(" 6. R/V диаграмма для пачки прямоугольных импульсов")
        
        self.plot_lfm_burst_signal()
        print(" 7. Пачка ЛЧМ сигналов + 3D ФН + R/V диаграмма")
        
        print("\n" + "="*70)
        print(" ВСЕ ГРАФИКИ УСПЕШНО ПОСТРОЕНЫ")
        print(" Добавлено задание 7: Пачка ЛЧМ сигналов")
        print(" Параметры пачки ЛЧМ: τи=2 мкс, Tпр=5 мкс, N=4, Δf=10 МГц")
        print(" Используйте ползунок справа для прокрутки")
        print(" 3D графики можно вращать мышью")
        print("="*70 + "\n")
        
    def run(self):
        """Запуск приложения"""
        self.root.mainloop()

# Запуск приложения
if __name__ == "__main__":
    app = RadarSignalApp()
    app.run()