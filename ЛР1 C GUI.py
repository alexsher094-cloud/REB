import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
Ta = 50e-6          # Время анализа, с
c0 = 3e8            # Скорость света, м/с
lambda_c = 0.03     # Длина волны, м (для f=10 ГГц)
f0 = 10e9           # Рабочая частота РЛС, Гц

# Вектор времени
t = np.arange(0, Ta, Ts)
Nt = len(t)
f = np.arange(Nt) * Fs / Nt

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

# ==================== ГЛАВНОЕ ПРИЛОЖЕНИЕ ====================
class RadarSignalApp:
    def __init__(self):
        self.root = ctk.CTk()
        self.root.title("Радиолокационные сигналы - Анализ функции неопределенности")
        self.root.geometry("1600x900")
        
        # Переменные для хранения результатов
        self.current_figure = None
        self.canvas = None
        self.results_text = ""
        
        # Настройка сетки (исправлено: columnconfigure вместо grid_column_configure)
        self.root.columnconfigure(0, weight=3)  # Графики
        self.root.columnconfigure(1, weight=1)  # Панель информации
        self.root.rowconfigure(0, weight=1)
        
        # Создание виджетов
        self.setup_info_panel()
        self.setup_main_frame()
        self.setup_buttons()
        
        # Перенаправление stdout для вывода в GUI
        self.setup_output_redirect()
        
        # Вывод приветствия
        self.print_welcome()
        
    def print_welcome(self):
        """Вывод приветственного сообщения"""
        print("\n РАДИОЛОКАЦИОННЫЕ СИГНАЛЫ - АНАЛИЗ ФУНКЦИИ НЕОПРЕДЕЛЕННОСТИ")
        print("\n Выберите тип сигнала из меню ниже для анализа")
        print("\n Все результаты выводятся в эту панель")
        print("\n Графики отображаются слева\n")
        print("\nДоступные сигналы:")
        print(" \n 1. Прямоугольный импульс (τи = 0.25 мкс)")
        print(" \n 2. Сигнал с ЛЧМ (τи = 1 мкс, Δf = 20 МГц)")
        print(" \n 3. Код Баркера (длина 13)")
        print(" \n 4. М-последовательность (длина 15)")
        print(" \n 5. Пачка импульсов (4 импульса)")
        print(" \n 6. R/V диаграмма для пачки импульсов\n")
        
    def setup_info_panel(self):
        """Правая панель с информацией"""
        self.info_frame = ctk.CTkFrame(self.root, width=350, corner_radius=10)
        self.info_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.info_frame.grid_propagate(False)
        
        # Заголовок
        title_label = ctk.CTkLabel(self.info_frame, text=" ИНФОРМАЦИЯ", 
                                    font=ctk.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=10)
        
        # Текстовое поле для вывода
        self.output_text = ctk.CTkTextbox(self.info_frame, width=330, height=400, 
                                           font=ctk.CTkFont(size=11))
        self.output_text.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Разделитель
        separator = ctk.CTkFrame(self.info_frame, height=2, fg_color="gray")
        separator.pack(fill="x", padx=10, pady=5)
        
        # Параметры сигнала
        params_label = ctk.CTkLabel(self.info_frame, text=" ТЕКУЩИЙ СИГНАЛ", 
                                     font=ctk.CTkFont(size=14, weight="bold"))
        params_label.pack(pady=5)
        
        self.params_frame = ctk.CTkFrame(self.info_frame)
        self.params_frame.pack(pady=5, padx=10, fill="x")
        
        self.current_signal_label = ctk.CTkLabel(self.params_frame, text="Не выбран", 
                                                  font=ctk.CTkFont(size=12))
        self.current_signal_label.pack(pady=2)
        
        self.resolution_label = ctk.CTkLabel(self.params_frame, text="", 
                                              font=ctk.CTkFont(size=11))
        self.resolution_label.pack(pady=2)
        
        # Кнопка очистки
        clear_btn = ctk.CTkButton(self.info_frame, text="Очистить вывод", 
                                   command=self.clear_output, height=30)
        clear_btn.pack(pady=10, padx=20, fill="x")
        
    def setup_main_frame(self):
        """Основная область для графиков"""
        self.main_frame = ctk.CTkFrame(self.root, corner_radius=10)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Заголовок
        title_label = ctk.CTkLabel(self.main_frame, text=" ФУНКЦИЯ НЕОПРЕДЕЛЕННОСТИ РАДИОСИГНАЛОВ", 
                                    font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=10)
        
        # Frame для canvas matplotlib
        self.canvas_frame = ctk.CTkFrame(self.main_frame)
        self.canvas_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
    def setup_buttons(self):
        """Кнопки управления внизу"""
        button_frame = ctk.CTkFrame(self.root, height=60, corner_radius=10)
        button_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        # Ряд кнопок
        buttons = [
            (" Прямоугольный импульс", self.calc_pulse),
            (" Сигнал с ЛЧМ", self.calc_lfm),
            (" Код Баркера", self.calc_barker),
            (" М-последовательность", self.calc_msequence),
            (" Пачка импульсов", self.calc_burst),
            (" R/V график", self.calc_rv_plot)
        ]
        
        for i, (text, command) in enumerate(buttons):
            btn = ctk.CTkButton(button_frame, text=text, command=command, 
                                 width=140, height=35, font=ctk.CTkFont(size=12))
            btn.grid(row=0, column=i, padx=5, pady=10)
            
        # Настройка колонок для равномерного распределения
        for i in range(len(buttons)):
            button_frame.columnconfigure(i, weight=1)
            
    def setup_output_redirect(self):
        """Перенаправление stdout в текстовое поле"""
        class OutputRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
                
            def write(self, text):
                if text.strip():
                    self.text_widget.insert("end", text)
                    self.text_widget.see("end")
                    
            def flush(self):
                pass
                
        sys.stdout = OutputRedirector(self.output_text)
        
    def clear_output(self):
        """Очистка текстового поля"""
        self.output_text.delete("1.0", "end")
        self.print_welcome()
        
    def update_info_panel(self, signal_name, params, delta_R, delta_V):
        """Обновление панели информации"""
        self.current_signal_label.configure(text=f" {signal_name}")
        info_text = f"τи = {params.get('Tu', 0)*1e6:.2f} мкс\n"
        if 'deltaF' in params:
            info_text += f"Δf = {params['deltaF']/1e6:.0f} МГц\n"
        if 'code_len' in params:
            info_text += f"Длина кода = {params['code_len']}\n"
        if 'N_pt' in params:
            info_text += f"N = {params['N_pt']}\n"
            info_text += f"Tпр = {params.get('Tpr', 0)*1e6:.2f} мкс\n"
        info_text += f"\nРазрешение:\n"
        info_text += f"По дальности: {delta_R:.1f} м\n"
        info_text += f"По скорости: {delta_V:.2f} м/с"
        self.resolution_label.configure(text=info_text)
        
    def plot_figure(self, fig):
        """Отображение figure matplotlib в GUI"""
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        
    def calc_pulse(self):
        """Задание 1: Одиночный прямоугольный импульс"""
        print("\n")
        print("\nЗАДАНИЕ 1: Одиночный прямоугольный импульс")
        print("\n")
        
        
        Tu = 0.25e-6
        Tpr = 1.5e-6
        N_pt = 1
        
        S_ampl = np.zeros(Nt, dtype=float)
        for n in range(N_pt):
            mask = (t >= n * Tpr) & (t < n * Tpr + Tu)
            S_ampl[mask] = 1.0
        
        U = S_ampl * np.exp(1j * 2 * np.pi * IF * t)
        S = fft(U)
        B = correlate(U, U, mode='full')
        tau = np.linspace(-Ta, Ta, len(B))
        
        tau_min, tau_max = -Tu, Tu
        f_max_display = 40e6
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, 200, f_max_display)
        
        # Построение графиков
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle('Прямоугольный импульс (τи = 0.25 мкс)', fontsize=14, color='white')
        fig.patch.set_facecolor('#2b2b2b')
        
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
        axes[0, 1].set_xlabel('f - f₀, МГц')
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
        contour1 = axes[1, 1].contour(f_plot, tau_af*1e6, amf, contour_levels, colors='white', linewidths=1.5)
        axes[1, 1].clabel(contour1, inline=True, fontsize=8, colors='white')
        axes[1, 1].set_xlabel('f - f₀, МГц')
        axes[1, 1].set_ylabel('τ, мкс')
        axes[1, 1].set_title('Сечения ФН')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        self.plot_figure(fig)
        
        delta_R = c0 * Tu / 2
        delta_V = (1/Tu) * lambda_c / 2
        self.update_info_panel("Прямоугольный импульс", {'Tu': Tu, 'Tpr': Tpr, 'N_pt': N_pt}, delta_R, delta_V)
        
        print(f" Разрешение по дальности: {delta_R:.2f} м")
        print(f" Разрешение по скорости: {delta_V:.2f} м/с")
        print(f" Ширина спектра: ~{1/Tu/1e6:.1f} МГц")
        
    def calc_lfm(self):
        """Задание 2: Сигнал с ЛЧМ"""
        print("\n")
        print("\nЗАДАНИЕ 2: Сигнал с ЛЧМ")
        print("\n")
        
        
        Tu = 1e-6
        Tpr = 5e-6
        N_pt = 1
        deltaF = 20e6
        
        S_ampl = np.zeros(Nt, dtype=float)
        for n in range(N_pt):
            mask = (t >= n * Tpr) & (t < n * Tpr + Tu)
            S_ampl[mask] = 1.0
        
        U = np.zeros(Nt, dtype=complex)
        for n in range(N_pt):
            mask = (t >= n * Tpr) & (t < n * Tpr + Tu)
            t_local = t[mask] - n * Tpr
            phase = 2 * np.pi * IF * t_local + 2 * np.pi * (deltaF/(2*Tu)) * t_local**2
            U[mask] = S_ampl[mask] * np.exp(1j * phase)
        
        S = fft(U)
        B = correlate(U, U, mode='full')
        tau = np.linspace(-Ta, Ta, len(B))
        
        tau_min, tau_max = -Tu, Tu
        f_max_display = 25e6
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, 200, f_max_display)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(f'Сигнал с ЛЧМ (τи={Tu*1e6:.2f} мкс, Δf={deltaF/1e6:.0f} МГц)', fontsize=14, color='white')
        fig.patch.set_facecolor('#2b2b2b')
        
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
        axes[0, 1].set_xlabel('f - f₀, МГц')
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
        axes[1, 1].set_xlabel('f - f₀, МГц')
        axes[1, 1].set_ylabel('τ, мкс')
        axes[1, 1].set_title('ФН (дБ) - синий:0 дБ, красный:много дБ')
        plt.colorbar(surf, ax=axes[1, 1], label='дБ')
        
        plt.tight_layout()
        self.plot_figure(fig)
        
        delta_R = c0 / (2 * deltaF)
        delta_V = (1/Tu) * lambda_c / 2
        self.update_info_panel("Сигнал с ЛЧМ", {'Tu': Tu, 'deltaF': deltaF}, delta_R, delta_V)
        
        print(f" Разрешение по дальности (после сжатия): {delta_R:.2f} м")
        print(f" Разрешение по скорости: {delta_V:.2f} м/с")
        print(f" База сигнала: {Tu * deltaF:.0f}")
        
    def calc_barker(self):
        """Задание 3: Код Баркера"""
        print("\n")
        print("\nЗАДАНИЕ 3: Код Баркера (длина 13)")
        print("\n")
        
        
        code = [1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1]
        Tu = 0.25e-6
        Tpr = Tu * len(code)
        
        S_ampl = np.zeros(Nt, dtype=float)
        for i, val in enumerate(code):
            mask = (t >= i * Tu) & (t < i * Tu + Tu)
            S_ampl[mask] = val
        
        U = S_ampl * np.exp(1j * 2 * np.pi * IF * t)
        S = fft(U)
        B = correlate(U, U, mode='full')
        tau = np.linspace(-Ta, Ta, len(B))
        
        tau_min, tau_max = -10*Tu, 10*Tu
        f_max_display = 8e6
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, 300, f_max_display)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(f'Код Баркера (длина {len(code)}, τэ={Tu*1e6:.2f} мкс)', fontsize=14, color='white')
        fig.patch.set_facecolor('#2b2b2b')
        
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
        axes[0, 1].set_xlabel('f - f₀, МГц')
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
        contour3 = axes[1, 1].contour(f_plot, tau_af*1e6, amf, contour_levels, colors='white', linewidths=1.5)
        axes[1, 1].clabel(contour3, inline=True, fontsize=8, colors='white')
        axes[1, 1].set_xlabel('f - f₀, МГц')
        axes[1, 1].set_ylabel('τ, мкс')
        axes[1, 1].set_title('Сечения ФН')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.plot_figure(fig)
        
        delta_R = c0 * Tu / 2
        delta_V = (1/(len(code)*Tu)) * lambda_c / 2
        self.update_info_panel("Код Баркера", {'Tu': Tu, 'code_len': len(code)}, delta_R, delta_V)
        
        print(f" Разрешение по дальности: {delta_R:.2f} м")
        print(f" Разрешение по скорости: {delta_V:.2f} м/с")
        print(f" База сигнала: {len(code)}")
        print(f" Уровень боковых лепестков АКФ: 1/{len(code)} = {1/len(code):.3f} ({20*np.log10(1/len(code)):.1f} дБ)")
        
    def calc_msequence(self):
        """Задание 4: М-последовательность"""
        print("\n")
        print("\nЗАДАНИЕ 4: М-последовательность (длина 15)")
        print("\n")
        
        
        code = [1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1]
        Tu = 0.25e-6
        Tpr = Tu * len(code)
        
        S_ampl = np.zeros(Nt, dtype=float)
        for i, val in enumerate(code):
            mask = (t >= i * Tu) & (t < i * Tu + Tu)
            S_ampl[mask] = val
        
        U = S_ampl * np.exp(1j * 2 * np.pi * IF * t)
        S = fft(U)
        B = correlate(U, U, mode='full')
        tau = np.linspace(-Ta, Ta, len(B))
        
        tau_min, tau_max = -10*Tu, 10*Tu
        f_max_display = 8e6
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, 300, f_max_display)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(f'М-последовательность (длина {len(code)}, τэ={Tu*1e6:.2f} мкс)', fontsize=14, color='white')
        fig.patch.set_facecolor('#2b2b2b')
        
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
        axes[0, 1].set_xlabel('f - f₀, МГц')
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
        contour4 = axes[1, 1].contour(f_plot, tau_af*1e6, amf, contour_levels, colors='white', linewidths=1.5)
        axes[1, 1].clabel(contour4, inline=True, fontsize=8, colors='white')
        axes[1, 1].set_xlabel('f - f₀, МГц')
        axes[1, 1].set_ylabel('τ, мкс')
        axes[1, 1].set_title('Сечения ФН')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.plot_figure(fig)
        
        delta_R = c0 * Tu / 2
        delta_V = (1/(len(code)*Tu)) * lambda_c / 2
        self.update_info_panel("М-последовательность", {'Tu': Tu, 'code_len': len(code)}, delta_R, delta_V)
        
        print(f" Разрешение по дальности: {delta_R:.2f} м")
        print(f" Разрешение по скорости: {delta_V:.2f} м/с")
        print(f" База сигнала: {len(code)}")
        
    def calc_burst(self):
        """Задание 5: Пачка импульсов"""
        print("\n")
        print("\nЗАДАНИЕ 5: Пачка прямоугольных импульсов")
        print("\n")
        
        
        Tu = 0.25e-6
        Tpr = 1.5e-6
        N_pt = 4
        
        S_ampl = np.zeros(Nt, dtype=float)
        for n in range(N_pt):
            mask = (t >= n * Tpr) & (t < n * Tpr + Tu)
            S_ampl[mask] = 1.0
        
        U = S_ampl * np.exp(1j * 2 * np.pi * IF * t)
        S = fft(U)
        B = correlate(U, U, mode='full')
        tau = np.linspace(-Ta, Ta, len(B))
        
        tau_min, tau_max = -8*Tu, 8*Tu
        f_max_display = 40e6
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, 300, f_max_display)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle(f'Пачка импульсов (N={N_pt}, Tпр={Tpr*1e6:.2f} мкс)', fontsize=14, color='white')
        fig.patch.set_facecolor('#2b2b2b')
        
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
        axes[0, 1].set_xlabel('f - f₀, МГц')
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
        contour5 = axes[1, 1].contour(f_plot, tau_af*1e6, amf, contour_levels, colors='white', linewidths=1.5)
        axes[1, 1].clabel(contour5, inline=True, fontsize=8, colors='white')
        axes[1, 1].set_xlabel('f - f₀, МГц')
        axes[1, 1].set_ylabel('τ, мкс')
        axes[1, 1].set_title('Сечения ФН')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        self.plot_figure(fig)
        
        delta_R = c0 * Tu / 2
        delta_V = c0 / (2 * f0 * N_pt * Tpr)
        self.update_info_panel("Пачка импульсов", {'Tu': Tu, 'Tpr': Tpr, 'N_pt': N_pt}, delta_R, delta_V)
        
        print(f" Разрешение по дальности (одиночный импульс): {delta_R:.2f} м")
        print(f" Разрешение по скорости (пачка): {delta_V:.2f} м/с")
        print(f" Период неоднозначности по дальности: {c0 * Tpr / 2:.1f} м")
        
    def calc_rv_plot(self):
        """Дополнительный график: ФН в координатах (R, V)"""
        print("\n")
        print("\nДОПОЛНИТЕЛЬНЫЙ ГРАФИК: R/V диаграмма для пачки импульсов")
        print("\n")
        
        
        Tu = 0.25e-6
        Tpr = 1.5e-6
        N_pt = 4
        
        S_ampl = np.zeros(Nt, dtype=float)
        for n in range(N_pt):
            mask = (t >= n * Tpr) & (t < n * Tpr + Tu)
            S_ampl[mask] = 1.0
        
        U = S_ampl * np.exp(1j * 2 * np.pi * IF * t)
        
        tau_min, tau_max = -8*Tu, 8*Tu
        f_max_display = 40e6
        amf, tau_af, f_plot = ambiguity_function_full(U, t, Ts, tau_min, tau_max, 300, f_max_display)
        
        # Пересчет в координаты дальности и скорости
        R_tau = tau_af * c0 / 2
        V_f = f_plot * 1e6 * c0 / (2 * f0)
        
        # Физические пределы
        R_max_unambiguous = c0 * Tpr / 2
        delta_R = c0 * Tu / 2
        delta_V = c0 / (2 * f0 * N_pt * Tpr)
        
        R_limit = min(3 * R_max_unambiguous, 1000)
        V_limit = 3 * c0 / (2 * f0 * Tpr)
        
        mask_R = np.abs(R_tau) <= R_limit
        mask_V = np.abs(V_f) <= V_limit
        R_tau_cut = R_tau[mask_R]
        V_f_cut = V_f[mask_V]
        amf_cut = amf[np.ix_(mask_R, mask_V)]
        
        print(f" Период неоднозначности по дальности: {R_max_unambiguous:.1f} м")
        print(f" Разрешение по дальности: {delta_R:.1f} м")
        print(f" Разрешение по скорости: {delta_V:.2f} м/с")
        print(f" Пределы графика: R = ±{R_limit:.0f} м, V = ±{V_limit:.0f} м/с")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('#2b2b2b')
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
        ax.set_title('Функция неопределенности пачки импульсов\nв координатах (дальность, скорость)', fontsize=14)
        
        ax.axhline(y=0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.set_xlim([-V_limit, V_limit])
        ax.set_ylim([-R_limit, R_limit])
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.text(0, 0, 'Главный\nлепесток', ha='center', va='center', 
                fontsize=10, color='white', fontweight='bold')
        
        param_text = f'Параметры сигнала:\n'
        param_text += f'τи = {Tu*1e6:.2f} мкс\n'
        param_text += f'Tпр = {Tpr*1e6:.2f} мкс\n'
        param_text += f'N = {N_pt}\n\n'
        param_text += f'ΔR = {delta_R:.1f} м\n'
        param_text += f'ΔV = {delta_V:.2f} м/с\n'
        param_text += f'R_max = {R_max_unambiguous:.0f} м'
        
        ax.annotate(param_text, xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=10, va='top', ha='left', color='black',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        self.plot_figure(fig)
        
    def run(self):
        """Запуск приложения"""
        self.root.mainloop()

# Запуск приложения
if __name__ == "__main__":
    app = RadarSignalApp()
    app.run()
