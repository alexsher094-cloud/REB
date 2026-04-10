import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import fftconvolve
from scipy.fft import fft, fftshift, ifft

# ==========================================
# ОБЩИЕ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def add_awgn(signal, snr_db):
    """Добавление белого гауссовского шума"""
    signal_power = np.mean(np.abs(signal)**2)
    if signal_power == 0: return signal # Защита от тишины
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise

# ==========================================
# КЛАСС ГЛАВНОГО ПРИЛОЖЕНИЯ
# ==========================================

class RadarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Radar Signal Simulation Suite")
        self.root.geometry("1200x800")

        # Создаем вкладки (Notebook)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)

        # Вкладка 1: ЛЧМ
        self.tab_lfm = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_lfm, text='1. ЛЧМ Сигнал')
        self.setup_lfm_tab()

        # Вкладка 2: Баркер
        self.tab_barker = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_barker, text='2. Код Баркера')
        self.setup_barker_tab()

        # Вкладка 3: Произвольный код (M-последовательность)
        self.tab_custom = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_custom, text='3. Произвольный ФКМ')
        self.setup_custom_tab()

    # ==========================================
    # ЛОГИКА И ИНТЕРФЕЙС ВКЛАДКИ 1: ЛЧМ
    # ==========================================
    def setup_lfm_tab(self):
        # Панель управления
        control_frame = ttk.LabelFrame(self.tab_lfm, text="Параметры ЛЧМ", padding=10)
        control_frame.pack(side='left', fill='y', padx=5, pady=5)

        # Поля ввода
        self.entries_lfm = {}
        params = [
            ("Длит. импульса (мкс)", "10"),
            ("Период следования (мкс)", "100"),
            ("Кол-во импульсов (шт)", "1"),
            ("Девиация (МГц)", "10"),
            ("Пром. частота (МГц)", "50"),
            ("Несущая (МГц)", "200"),
            ("Дискретизация (МГц)", "400")
        ]

        for i, (label, default) in enumerate(params):
            ttk.Label(control_frame, text=label).grid(row=i, column=0, sticky='w', pady=2)
            entry = ttk.Entry(control_frame, width=10)
            entry.insert(0, default)
            entry.grid(row=i, column=1, padx=5, pady=2)
            self.entries_lfm[label] = entry

        btn_run = ttk.Button(control_frame, text="Запустить моделирование", command=self.run_lfm_simulation)
        btn_run.grid(row=len(params), column=0, columnspan=2, pady=15)

        # Область для графиков
        self.plot_frame_lfm = ttk.Frame(self.tab_lfm)
        self.plot_frame_lfm.pack(side='right', fill='both', expand=True)

    def run_lfm_simulation(self):
        try:
            # Считывание данных
            tau_u = float(self.entries_lfm["Длит. импульса (мкс)"].get()) * 1e-6
            T_sl = float(self.entries_lfm["Период следования (мкс)"].get()) * 1e-6
            N_u = int(float(self.entries_lfm["Кол-во импульсов (шт)"].get()))
            delta_f = float(self.entries_lfm["Девиация (МГц)"].get()) * 1e6
            f_if = float(self.entries_lfm["Пром. частота (МГц)"].get()) * 1e6
            f_c = float(self.entries_lfm["Несущая (МГц)"].get()) * 1e6
            f_s = float(self.entries_lfm["Дискретизация (МГц)"].get()) * 1e6
        except ValueError:
            messagebox.showerror("Ошибка", "Пожалуйста, введите корректные числовые значения.")
            return

        if T_sl <= tau_u:
            messagebox.showerror("Ошибка", "Период следования должен быть больше длительности импульса!")
            return

        # --- Математическая модель ЛЧМ ---
        def generate_lfm_pulse(tau, f_if, delta_f, f_s):
            t = np.arange(0, tau, 1/f_s)
            k = delta_f / tau
            phase = 2 * np.pi * f_if * t + np.pi * k * t**2
            return np.exp(1j * phase), t

        def matched_filter(rx_signal, ref_signal):
            N = len(rx_signal)
            ref_padded = np.zeros(N, dtype=complex)
            if len(ref_signal) > N:
                ref_signal = ref_signal[:N]
            ref_padded[:len(ref_signal)] = ref_signal
            Rx = np.fft.fft(rx_signal)
            H = np.conj(np.fft.fft(ref_padded))
            return np.fft.ifft(Rx * H)

        # 1. Генерация
        tx_pulse, _ = generate_lfm_pulse(tau_u, f_if, delta_f, f_s)
        samples_per_period = int(T_sl * f_s)
        total_samples = int(N_u * T_sl * f_s)
        tx_signal_full = np.zeros(total_samples, dtype=complex)
        
        for i in range(N_u):
            start_idx = i * samples_per_period
            end_idx = start_idx + len(tx_pulse)
            if end_idx <= total_samples:
                tx_signal_full[start_idx:end_idx] = tx_pulse
        
        t_full = np.arange(0, total_samples / f_s, 1/f_s)
        view_duration = max(3 * T_sl, min(N_u * T_sl, 5 * T_sl)) * 1e6 
        if N_u == 1: view_duration = tau_u * 2 * 1e6

        # --- Построение графиков ---
        for widget in self.plot_frame_lfm.winfo_children():
            widget.destroy()
            
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        gs = fig.add_gridspec(3, 2)
        
        # 1.1 Временная область
        ax1 = fig.add_subplot(gs[0, 0])
        rx_for_spectrum = add_awgn(tx_signal_full, 20)
        ax1.plot(t_full * 1e6, np.real(rx_for_spectrum), color='black')
        ax1.set_title('Временная область (Реальная часть, ОСШ=20 дБ)')
        ax1.set_xlabel('Время, мкс')
        ax1.set_ylabel('Амплитуда')
        ax1.grid(True)
        ax1.set_xlim(0, view_duration)

        # 1.2 Спектральная область
        ax2 = fig.add_subplot(gs[0, 1])
        spectrum = np.fft.fftshift(np.fft.fft(rx_for_spectrum))
        freqs = np.fft.fftshift(np.fft.fftfreq(total_samples, 1/f_s)) / 1e6
        ax2.plot(freqs, 20 * np.log10(np.abs(spectrum) / np.max(np.abs(spectrum))))
        ax2.set_title('Спектр пачки')
        ax2.set_xlabel('Частота, МГц')
        ax2.set_ylabel('Амплитуда, дБ')
        ax2.grid(True)
        ax2.set_xlim(f_if/1e6 - delta_f/1e6, f_if/1e6 + delta_f*2/1e6)

        # 1.3 Выход СФ (ОСШ)
        ax3 = fig.add_subplot(gs[1, :])
        snr_levels = [20, 0, -10]
        colors = ['green', 'blue', 'red']
        for i, snr in enumerate(snr_levels):
            rx_noisy = add_awgn(tx_signal_full, snr)
            mf_out = matched_filter(rx_noisy, tx_pulse)
            mf_out_db = 20 * np.log10(np.abs(mf_out) / np.max(np.abs(mf_out)))
            t_mf = np.arange(0, len(mf_out)/f_s, 1/f_s) * 1e6 
            ax3.plot(t_mf, mf_out_db, color=colors[i], label=f'ОСШ = {snr} дБ', alpha=0.8)
        
        ax3.set_title('Выход СФ (Настроен на 1 импульс)')
        ax3.set_xlabel('Время, мкс')
        ax3.set_ylabel('Амплитуда, дБ')
        ax3.grid(True)
        ax3.set_xlim(0, view_duration)
        ax3.set_ylim(-40, 5)
        ax3.legend()

        # 1.4 Доплер
        ax4 = fig.add_subplot(gs[2, :])
        fd_levels = [0, 1 / (4 * tau_u), 1 / tau_u]
        fd_labels = [f'Fd = 0 Гц', f'Fd = {fd_levels[1]/1e3:.1f} кГц', f'Fd = {fd_levels[2]/1e3:.1f} кГц']
        
        for i, fd in enumerate(fd_levels):
            rx_doppler_pulse, _ = generate_lfm_pulse(tau_u, f_if + fd, delta_f, f_s)
            rx_doppler_demo = np.zeros(samples_per_period, dtype=complex)
            rx_doppler_demo[:len(rx_doppler_pulse)] = rx_doppler_pulse
            
            mf_out_doppler = matched_filter(rx_doppler_demo, tx_pulse)
            mf_out_doppler_db = 20 * np.log10(np.abs(mf_out_doppler) / np.max(np.abs(mf_out_doppler)))
            t_mf_demo = np.arange(0, len(mf_out_doppler)/f_s, 1/f_s) * 1e6
            ax4.plot(t_mf_demo, mf_out_doppler_db, color=colors[i], label=fd_labels[i], linewidth=1.5 if fd==0 else 1)

        ax4.set_title('Влияние Доплера (на примере одного импульса)')
        ax4.set_xlabel('Время, мкс')
        ax4.set_ylabel('Амплитуда, дБ (нормир.)')
        ax4.grid(True)
        ax4.set_xlim(-tau_u*1.2*1e6, tau_u*1.2*1e6) 
        ax4.set_ylim(-40, 5)
        ax4.legend()

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame_lfm)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    # ==========================================
    # ЛОГИКА И ИНТЕРФЕЙС ВКЛАДКИ 2: БАРКЕР
    # ==========================================
    def setup_barker_tab(self):
        control_frame = ttk.LabelFrame(self.tab_barker, text="Параметры Баркера", padding=10)
        control_frame.pack(side='left', fill='y', padx=5, pady=5)

        # Выбор кода
        ttk.Label(control_frame, text="Тип кода:").pack(anchor='w', pady=5)
        self.code_type_var = tk.StringVar(value="std")
        ttk.Radiobutton(control_frame, text="Стандартный", variable=self.code_type_var, value="std", command=self.toggle_barker_inputs).pack(anchor='w')
        ttk.Radiobutton(control_frame, text="Свой код", variable=self.code_type_var, value="custom", command=self.toggle_barker_inputs).pack(anchor='w')

        # Длина стандартного кода
        self.frame_std_len = ttk.Frame(control_frame)
        self.frame_std_len.pack(anchor='w', pady=5)
        ttk.Label(self.frame_std_len, text="Длина:").pack(side='left')
        self.combo_barker_len = ttk.Combobox(self.frame_std_len, values=[3, 5, 7, 11, 13], width=5)
        self.combo_barker_len.current(4) # 13
        self.combo_barker_len.pack(side='left', padx=5)

        # Свой код
        self.frame_custom_code = ttk.Frame(control_frame)
        # ttk.Label(self.frame_custom_code, text="Код (1, -1):").pack(anchor='w')
        self.entry_custom_barker = ttk.Entry(self.frame_custom_code, width=20)
        self.entry_custom_barker.insert(0, "1, 1, -1, 1")
        self.entry_custom_barker.pack(pady=5)
        self.frame_custom_code.pack_forget() # Скрыт по умолчанию

        # Длительность
        ttk.Label(control_frame, text="Длит. импульса (мкс):").pack(anchor='w', pady=(10, 0))
        self.entry_barker_tau = ttk.Entry(control_frame, width=10)
        self.entry_barker_tau.insert(0, "13")
        self.entry_barker_tau.pack(pady=5)

        btn_run = ttk.Button(control_frame, text="Запустить моделирование", command=self.run_barker_simulation)
        btn_run.pack(pady=15)

        self.plot_frame_barker = ttk.Frame(self.tab_barker)
        self.plot_frame_barker.pack(side='right', fill='both', expand=True)

    def toggle_barker_inputs(self):
        if self.code_type_var.get() == "std":
            self.frame_std_len.pack(anchor='w', pady=5)
            self.frame_custom_code.pack_forget()
        else:
            self.frame_std_len.pack_forget()
            self.frame_custom_code.pack(anchor='w', pady=5)

    def run_barker_simulation(self):
        try:
            tau = float(self.entry_barker_tau.get()) * 1e-6
            
            # Получаем код
            if self.code_type_var.get() == "std":
                length = int(self.combo_barker_len.get())
                codes = {3: [1, 1, -1], 5: [1, 1, 1, -1, 1], 7: [1, 1, 1, -1, -1, 1, -1],
                         11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
                         13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]}
                code = np.array(codes.get(length, [1, 1, 1]))
            else:
                raw = self.entry_custom_barker.get()
                code = np.array([int(x.strip()) for x in raw.replace(',', ' ').split() if x.strip() in ['1', '-1']])
                if len(code) == 0: raise ValueError("Пустой код")

        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные параметры. Проверьте ввод.")
            return

        # --- Математическая модель Баркера ---
        # Параметры среды (фиксированные для демонстрации, как в оригинале)
        f_c = 200e6
        f_if = 50e6
        fs = 400e6
        T_rep = 100e-6
        N_pulses = 5
        target_delay = 2 * T_rep

        T_chip = tau / len(code)
        num_samples_chip = int(np.round(T_chip * fs))
        t_single = np.arange(0, T_rep, 1/fs)
        
        # Генерация импульса
        t_chip = np.arange(0, num_samples_chip / fs, 1/fs)
        carrier = np.exp(1j * 2 * np.pi * f_if * t_chip)
        pulse_bb = np.zeros(len(t_single), dtype=complex)
        current_idx = 0
        
        for bit in code:
            phase = 0 if bit > 0 else np.pi
            modulated_chip = np.exp(1j * phase) * carrier
            end_idx = current_idx + num_samples_chip
            if end_idx <= len(t_single):
                pulse_bb[current_idx:end_idx] = modulated_chip
                current_idx = end_idx
        
        # Генерация пачки
        total_samples = int(N_pulses * T_rep * fs)
        tx_burst = np.zeros(total_samples, dtype=complex)
        for i in range(N_pulses):
            start = i * len(t_single)
            end = start + len(t_single)
            if end <= total_samples:
                tx_burst[start:end] = pulse_bb
        
        ref_pulse = pulse_bb # Опорный сигнал

        def apply_channel(tx, snr_db, fd):
            delay_samp = int(target_delay * fs)
            rx = np.zeros_like(tx)
            if delay_samp < len(rx): rx[delay_samp:] = tx[:len(tx)-delay_samp]
            
            t = np.arange(len(rx)) / fs
            rx = rx * np.exp(1j * 2 * np.pi * fd * t)
            
            if snr_db is not None:
                sig_pow = np.mean(np.abs(rx)**2)
                noise_pow = sig_pow / (10**(snr_db/10))
                noise = np.sqrt(noise_pow/2) * (np.random.randn(len(rx)) + 1j*np.random.randn(len(rx)))
                rx += noise
            return rx

        # --- Графики ---
        for widget in self.plot_frame_barker.winfo_children():
            widget.destroy()

        fig = plt.Figure(figsize=(10, 8), dpi=100)
        
        # Подграфики для шума (3x3 сетка, упрощенная до 3 строк)
        # Строим только задачу 1 (Шум) и задачу 2 (Доплер) на разных фигурах или местах
        
        gs = fig.add_gridspec(3, 3)
        snr_levels = [20, 0, -10]

        for i, snr in enumerate(snr_levels):
            rx = apply_channel(tx_burst, snr, 0)
            mf_out = fftconvolve(rx, np.conj(ref_pulse[::-1]), mode='same')
            
            # Время
            ax = fig.add_subplot(gs[i, 0])
            st = int(target_delay * fs) - 100
            en = st + int(tau * fs) + 200
            ax.plot(np.arange(st, en)/fs * 1e6, np.real(rx[st:en]))
            ax.set_title(f'Сигнал (ОСШ={snr} дБ)')
            ax.set_xlabel('Время, мкс'); ax.grid(True)
            
            # Спектр
            ax = fig.add_subplot(gs[i, 1])
            N_fft = 8192
            spectrum = np.abs(fftshift(fft(rx, N_fft)))
            freqs = fftshift(np.fft.fftfreq(N_fft, 1/fs))
            msk = (freqs > f_if - 10e6) & (freqs < f_if + 10e6)
            ax.plot(freqs[msk]/1e6, 20*np.log10(spectrum[msk]+1e-12))
            ax.set_title('Спектр (ПЧ)')
            ax.set_xlabel('Частота, МГц'); ax.grid(True)
            
            # СФ
            ax = fig.add_subplot(gs[i, 2])
            ax.plot(np.arange(len(mf_out))/fs * 1e6, np.abs(mf_out))
            ax.set_title('Выход СФ')
            ax.set_xlabel('Время, мкс'); ax.grid(True)

        fig.suptitle(f'Баркер: Код {len(code)} эл., Tau={tau*1e6:.1f} мкс', fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame_barker)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    # ==========================================
    # ЛОГИКА И ИНТЕРФЕЙС ВКЛАДКИ 3: ПРОИЗВОЛЬНЫЙ КОД
    # ==========================================
    def setup_custom_tab(self):
        control_frame = ttk.LabelFrame(self.tab_custom, text="Параметры ФКМ (M-послед)", padding=10)
        control_frame.pack(side='left', fill='y', padx=5, pady=5)

        ttk.Label(control_frame, text="Длит. импульса (мкс):").pack(anchor='w')
        self.entry_custom_tau = ttk.Entry(control_frame, width=10)
        self.entry_custom_tau.insert(0, "0.25")
        self.entry_custom_tau.pack(pady=5)

        ttk.Label(control_frame, text="Код (через запятую):").pack(anchor='w', pady=(10,0))
        lbl_hint = ttk.Label(control_frame, text="Пример: 1, 1, -1, 1", font=("Arial", 8), foreground="gray")
        lbl_hint.pack(anchor='w')
        self.entry_custom_code_str = ttk.Entry(control_frame, width=25)
        self.entry_custom_code_str.insert(0, "1, -1, 1, -1, -1, -1, -1, 1, 1, -1")
        self.entry_custom_code_str.pack(pady=5)

        btn_run = ttk.Button(control_frame, text="Запустить моделирование", command=self.run_custom_simulation)
        btn_run.pack(pady=15)

        self.plot_frame_custom = ttk.Frame(self.tab_custom)
        self.plot_frame_custom.pack(side='right', fill='both', expand=True)

    def run_custom_simulation(self):
        try:
            Tu = float(self.entry_custom_tau.get()) * 1e-6
            code_str = self.entry_custom_code_str.get()
            # Парсинг кода с поддержкой отрицательных чисел
            code = np.array([int(x.strip()) for x in code_str.split(',') if x.strip().lstrip('-').isdigit()])
            if len(code) == 0: raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Проверьте ввод длительности и кода.")
            return

        # --- Математическая модель ---
        T_rep = 100e-6
        f_IF = 50e6
        Fs = 400e6
        Ts = 1 / Fs
        t = np.arange(0, T_rep, Ts)
        N = len(t)

        T_chip = Tu / len(code)
        samples_per_chip = int(T_chip / Ts)
        if samples_per_chip < 1: samples_per_chip = 1

        # Растягиваем код
        baseband_code = np.repeat(code, samples_per_chip)
        
        # Вычисляем точное количество отсчетов для импульса длительностью Tu
        total_samples_pulse = int(Tu / Ts)
        
        # ИСПРАВЛЕНИЕ: Добавляем padding (нули), чтобы длина кода совпадала с total_samples_pulse
        # Это необходимо, так как int(T_chip/Ts) дает усечение, и сумма длин чипов может быть меньше Tu
        if len(baseband_code) < total_samples_pulse:
            baseband_code = np.pad(baseband_code, (0, total_samples_pulse - len(baseband_code)))
        else:
            baseband_code = baseband_code[:total_samples_pulse]
            
        delay_samples = int(10e-6 / Ts) 
        signal_IF_complex = np.zeros(N, dtype=complex)
        
        # Формируем импульс с несущей
        # Используем срезы времени, соответствующие длине baseband_code
        start_idx = delay_samples
        end_idx = delay_samples + total_samples_pulse
        
        if end_idx <= N:
            t_pulse = t[start_idx:end_idx]
            # Умножаем код на несущую
            carrier_part = np.exp(1j * 2 * np.pi * f_IF * t_pulse)
            signal_IF_complex[start_idx:end_idx] = baseband_code * carrier_part

        # Эталонный сигнал (для СФ)
        ref_signal = np.zeros(N, dtype=complex)
        # Эталон должен быть идентичен переданному импульсу, но начинаться с 0 (без задержки)
        if total_samples_pulse <= N:
            t_ref = t[:total_samples_pulse]
            # Важно: используем тот же baseband_code (с padding)
            ref_signal[:total_samples_pulse] = baseband_code * np.exp(1j * 2 * np.pi * f_IF * t_ref)

        # Фильтр
        def matched_filter(rx, ref):
            h = np.conj(ref[::-1])
            n_conv = len(rx) + len(h) - 1
            Y = fft(rx, n=n_conv) * fft(h, n=n_conv)
            return ifft(Y)

        rx_static = signal_IF_complex.copy()
        signal_power = np.mean(np.abs(rx_static)**2)

        # --- Графики ---
        for widget in self.plot_frame_custom.winfo_children():
            widget.destroy()

        fig = plt.Figure(figsize=(10, 8), dpi=100)
        gs = fig.add_gridspec(3, 3)
        
        snr_values_db = [-5, 10, 20]

        for i, snr_db in enumerate(snr_values_db):
            # Шум
            snr_linear = 10**(snr_db / 10)
            noise_power = signal_power / snr_linear
            noise = (np.random.normal(0, 1, N) + 1j * np.random.normal(0, 1, N)) * np.sqrt(noise_power / 2)
            rx_noisy = rx_static + noise
            
            # 1. Время
            ax = fig.add_subplot(gs[i, 0])
            ax.plot(t * 1e6, np.real(rx_noisy))
            ax.set_title(f'Время (SNR={snr_db} дБ)')
            ax.set_xlabel('Время, мкс'); ax.grid(True)
            ax.set_xlim(0, 20)
            
            # 2. Спектр
            ax = fig.add_subplot(gs[i, 1])
            spectrum = fftshift(fft(rx_noisy))
            freq_axis = fftshift(np.fft.fftfreq(N, Ts))
            # Нормализация спектра для красоты
            spectrum_mag = 20 * np.log10(np.abs(spectrum) + 1e-12)
            # Центрируем по максимуму для лучшей визуализации
            max_val = np.max(spectrum_mag)
            
            ax.plot(freq_axis / 1e6, spectrum_mag)
            ax.set_title('Спектр')
            ax.set_xlabel('Частота, МГц'); ax.grid(True)
            ax.set_xlim(0, 100)
            ax.set_ylim(max_val - 60, max_val + 10) # Динамический диапазон
            
            # 3. СФ
            ax = fig.add_subplot(gs[i, 2])
            mf_out = matched_filter(rx_noisy, ref_signal)
            mf_out_mag = np.abs(mf_out)
            t_mf = np.arange(len(mf_out)) * Ts * 1e6
            
            ax.plot(t_mf, mf_out_mag)
            ax.set_title('СФ')
            ax.set_xlabel('Время, мкс'); ax.grid(True)
            
            peak_idx = np.argmax(mf_out_mag)
            center_us = t_mf[peak_idx]
            # Показываем окно вокруг пика
            ax.set_xlim(center_us - 2, center_us + 2)

        fig.suptitle(f'Произвольный код: {len(code)} эл., Tu={Tu*1e6:.2f} мкс', fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame_custom)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = RadarApp(root)
    root.mainloop()