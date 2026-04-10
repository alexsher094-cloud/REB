import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# ==========================================
# 1. КОНФИГУРАЦИЯ И ПАРАМЕТРЫ
# ==========================================

class RadarConfig:
    def __init__(self):
        # Статические общие данные
        self.f_c = 200e6          # Несущая частота (Гц)
        self.f_if = 50e6          # Промежуточная частота (Гц)
        self.fs = 400e6           # Частота дискретизации (Гц)
        self.T_rep = 100e-6       # Период следования импульсов (сек)
        
        # Параметры по умолчанию
        self.N_pulses = 5         # Количество импульсов в пачке
        self.code = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]) # Баркер-13 по умолчанию
        self.tau_p = 13e-6        # Длительность импульса (сек)
        self.delta_f = 1e6        # Девиация (будет пересчитана)
        
        # Параметры цели и среды
        self.target_delay = 2 * self.T_rep
        self.snr_levels_db = [20, 0, -10]
        self.doppler_shifts = [0, 50e3, 400e3]
        
        # Инициализация расчетных параметров
        self.recalc_params()

    def recalc_params(self):
        """Пересчет зависимых параметров."""
        code_length = len(self.code)
        self.T_chip = self.tau_p / code_length
        self.delta_f = 1 / self.T_chip
        
    def update_params(self, custom_code, tau_us):
        """Обновление параметров из ввода пользователя."""
        self.code = np.array(custom_code)
        self.tau_p = tau_us * 1e-6
        self.recalc_params()
        
        print(f"\n--- Принятые параметры ---")
        print(f"Код: {self.code}")
        print(f"Длина кода: {len(self.code)}")
        print(f"Длительность импульса: {self.tau_p*1e6:.2f} мкс")
        print(f"Длительность одного чипа: {self.T_chip*1e6:.2f} мкс")
        print(f"Расчетная полоса (Девиация): {self.delta_f/1e6:.2f} МГц")

# ==========================================
# 2. ГЕНЕРАТОРЫ СИГНАЛОВ
# ==========================================

def get_standard_barker_code(length):
    """Словарь стандартных кодов."""
    codes = {
        3: [1, 1, -1],
        5: [1, 1, 1, -1, 1],
        7: [1, 1, 1, -1, -1, 1, -1],
        11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
        13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
    }
    return codes.get(length, [1, 1, 1])

def generate_pulse_train(cfg):
    """Генерирует сигнал на основе cfg.code."""
    t_single = np.arange(0, cfg.T_rep, 1/cfg.fs)
    
    # Расчет отсчетов на чип
    num_samples_chip = int(np.round(cfg.T_chip * cfg.fs))
    
    # Базовый импульс
    t_chip = np.arange(0, num_samples_chip / cfg.fs, 1/cfg.fs)
    carrier = np.cos(2 * np.pi * cfg.f_if * t_chip)
    
    # Используем код из конфигурации
    pulse_bb = np.zeros(len(t_single), dtype=complex)
    current_idx = 0
    
    for bit in cfg.code:
        # phase = 0 для +1, phase = pi для -1
        phase = 0 if bit > 0 else np.pi
        modulated_chip = np.exp(1j * phase) * carrier
        
        end_idx = current_idx + num_samples_chip
        if end_idx <= len(t_single):
            pulse_bb[current_idx:end_idx] = modulated_chip
            current_idx = end_idx
    
    pulse_real = np.real(pulse_bb)
    
    # Формируем пачку
    total_samples = int(cfg.N_pulses * cfg.T_rep * cfg.fs)
    tx_signal = np.zeros(total_samples)
    
    for i in range(cfg.N_pulses):
        start = i * len(t_single)
        end = start + len(t_single)
        if end <= total_samples:
            tx_signal[start:end] = pulse_real
            
    return tx_signal, pulse_real

def apply_channel_effects(tx_signal, cfg, snr_db=None, fd=0):
    """Моделирует канал."""
    delay_samples = int(cfg.target_delay * cfg.fs)
    
    # Задержка
    rx_delayed = np.zeros_like(tx_signal)
    if delay_samples < len(rx_delayed):
        rx_delayed[delay_samples:] = tx_signal[:len(tx_signal)-delay_samples]
    
    # Доплер (моделирование сдвига частоты)
    t = np.arange(len(rx_delayed)) / cfg.fs
    doppler_carrier = np.cos(2 * np.pi * fd * t)
    rx_signal = rx_delayed * doppler_carrier
    
    # Шум
    if snr_db is not None:
        signal_power = np.mean(rx_signal**2)
        snr_linear = 10**(snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.normal(0, 1, len(rx_signal))
        rx_signal += noise
        
    return rx_signal

def matched_filter(rx_signal, reference_pulse):
    return fftconvolve(rx_signal, reference_pulse[::-1], mode='same')

# ==========================================
# 3. ВВОД ДАННЫХ (ТЕКСТОВЫЙ ИНТЕРФЕЙС)
# ==========================================

def get_user_code():
    print("\n" + "="*50)
    print(" ВЫБОР КОДА БАРКЕРА ")
    print("="*50)
    print("1. Использовать стандартный код (выбрать длину)")
    print("2. Ввести свой код вручную")
    
    choice = input("Ваш выбор (1 или 2): ").strip()
    
    code = []
    
    if choice == '1':
        while True:
            try:
                length = int(input("Введите длину стандартного кода (3, 5, 7, 11, 13): "))
                if length in [3, 5, 7, 11, 13]:
                    code = get_standard_barker_code(length)
                    print(f"Выбран стандартный код Баркера-{length}: {code}")
                    break
                else:
                    print("Некорректная длина. Попробуйте снова.")
            except ValueError:
                print("Введите число.")
                
    elif choice == '2':
        print("\nВведите код последовательностью 1 и -1 (или 0 вместо -1).")
        print("Пример: 1 1 -1 1 или 1,1,0,1")
        while True:
            raw_input = input("Код: ").strip()
            # Разделение по пробелам или запятым
            parts = raw_input.replace(',', ' ').split()
            
            try:
                temp_code = []
                valid = True
                for p in parts:
                    val = int(p)
                    # Преобразуем 0 в -1 для удобства (0 фаза и Пи фаза)
                    if val == 0: val = -1
                    
                    if val in [1, -1]:
                        temp_code.append(val)
                    else:
                        print(f"Ошибка: элемент '{val}' не равен 1, -1 или 0.")
                        valid = False
                        break
                
                if valid and len(temp_code) > 0:
                    code = temp_code
                    print(f"Принят пользовательский код: {code}")
                    break
                else:
                    print("Код не может быть пустым или содержит ошибки.")
            except ValueError:
                print("Ошибка ввода. Используйте целые числа.")
    else:
        print("Неверный выбор. Установлен код по умолчанию (Баркер-13).")
        code = get_standard_barker_code(13)
        
    return code

def get_user_input():
    code = get_user_code()
    
    while True:
        try:
            tau = float(input("Введите длительность импульса (микросекунды), напр. 10.0: "))
            if tau > 0:
                return code, tau
            print("Длительность должна быть > 0")
        except ValueError:
            print("Введите число.")

# ==========================================
# 4. ГЛАВНАЯ ЛОГИКА И ВИЗУАЛИЗАЦИЯ
# ==========================================

def run_simulation(cfg):
    # Генерация
    _, ref_pulse = generate_pulse_train(cfg)
    tx_burst, _ = generate_pulse_train(cfg)
    
    print(f"\nРасчет симуляции (Пачка из {cfg.N_pulses} импульсов)...")
    
    # --- ЗАДАЧА 1: ШУМ ---
    fig1, axes1 = plt.subplots(3, 3, figsize=(15, 10))
    fig1.suptitle(f'Задача 1: Шум. Код: {cfg.code}, $\tau$={cfg.tau_p*1e6:.1f} мкс', fontsize=14)
    
    for i, snr in enumerate(cfg.snr_levels_db):
        rx = apply_channel_effects(tx_burst, cfg, snr_db=snr, fd=0)
        mf_out = matched_filter(rx, ref_pulse)
        
        N_fft = 8192
        spectrum = np.abs(np.fft.fft(rx, N_fft))
        freqs = np.fft.fftfreq(N_fft, 1/cfg.fs)
        spectrum = np.fft.fftshift(spectrum)
        freqs = np.fft.fftshift(freqs)
        
        # 1. Время
        ax = axes1[i, 0]
        st = int(cfg.target_delay * cfg.fs) - 100
        en = st + int(cfg.tau_p * cfg.fs) + 200
        ax.plot(np.arange(st, en)/cfg.fs * 1e6, rx[st:en])
        ax.set_title(f'Сигнал (ОСШ={snr} дБ)')
        ax.set_xlabel('Время, мкс'); ax.grid(True)
        
        # 2. Спектр
        ax = axes1[i, 1]
        msk = (freqs > cfg.f_if - 10e6) & (freqs < cfg.f_if + 10e6)
        ax.plot(freqs[msk]/1e6, spectrum[msk])
        ax.set_title('Спектр (ПЧ)')
        ax.set_xlabel('Частота, МГц'); ax.grid(True)
        
        # 3. СФ
        ax = axes1[i, 2]
        t_mf = np.arange(int(cfg.N_pulses * cfg.T_rep * cfg.fs))/cfg.fs
        ax.plot(t_mf * 1e6, np.abs(mf_out)[:len(t_mf)])
        ax.set_title('Выход СФ')
        ax.set_xlabel('Время, мкс'); ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- ЗАДАЧА 2: ДОПЛЕР ---
    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 12))
    fig2.suptitle('Задача 2: Эффект Доплера (Без шума)', fontsize=14)
    
    ref_peak = 0
    for i, fd in enumerate(cfg.doppler_shifts):
        rx = apply_channel_effects(tx_burst, cfg, snr_db=None, fd=fd)
        mf_out = matched_filter(rx, ref_pulse)
        mf_env = np.abs(mf_out)
        
        ax = axes2[i]
        pk = np.argmax(mf_env)
        w = int(2 * cfg.tau_p * cfg.fs)
        sl = slice(max(0, pk-w), min(len(mf_env), pk+w))
        
        ax.plot(np.arange(sl.start, sl.stop)/cfg.fs * 1e6, mf_env[sl], label=f'fd={fd/1e3} кГц')
        
        peak_val = np.max(mf_env)
        if i == 0: ref_peak = peak_val
        loss = 20 * np.log10(peak_val / ref_peak) if ref_peak > 0 else 0
        
        ax.set_title(f'fd = {fd/1e3} кГц. Потери: {loss:.2f} дБ')
        ax.set_xlabel('Время, мкс'); ax.legend(); ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    cfg = RadarConfig()
    
    # Ввод пользователя
    user_code, user_tau = get_user_input()
    
    # Применение
    cfg.update_params(user_code, user_tau)
    
    # Запуск
    run_simulation(cfg)