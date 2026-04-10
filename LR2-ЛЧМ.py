import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# ФУНКЦИЯ ВВОДА ПАРАМЕТРОВ
# ==========================================
def get_user_input(prompt, default, unit, multiplier=1.0):
    """
    Запрашивает у пользователя ввод параметра.
    multiplier: коэффициент для перевода в единицы СИ
    """
    default_display = default / multiplier
    while True:
        try:
            user_val = input(f"{prompt} [{default_display:.1f} {unit}]: ")
            if not user_val.strip():
                return default
            return float(user_val) * multiplier
        except ValueError:
            print("Ошибка: введите корректное число.")

# ==========================================
# ВСПМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================

def generate_lfm_pulse(tau, f_if, delta_f, f_s):
    """Генерация одного ЛЧМ импульса"""
    t = np.arange(0, tau, 1/f_s)
    k = delta_f / tau
    phase = 2 * np.pi * f_if * t + np.pi * k * t**2
    return np.exp(1j * phase), t

def add_awgn(signal, snr_db):
    """Добавление белого гауссовского шума"""
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal)) + 1j*np.random.randn(len(signal)))
    return signal + noise

def matched_filter(rx_signal, ref_signal):
    """Согласованная фильтрация"""
    N = len(rx_signal)
    ref_padded = np.zeros(N, dtype=complex)
    # Если эталон длиннее сигнала (редкий случай), обрезаем эталон или сигнал
    if len(ref_signal) > N:
        ref_signal = ref_signal[:N]
        
    ref_padded[:len(ref_signal)] = ref_signal
    
    Rx = np.fft.fft(rx_signal)
    H = np.conj(np.fft.fft(ref_padded))
    return np.fft.ifft(Rx * H)

# ==========================================
# ЛОГИКА СИМУЛЯЦИИ
# ==========================================

def run_simulation(tau_u, T_sl, N_u, delta_f, f_c, f_if, f_s):
    # 1. Генерация одиночного импульса (эталон)
    tx_pulse, _ = generate_lfm_pulse(tau_u, f_if, delta_f, f_s)
    
    # 2. Формирование пачки импульсов
    # Вычисляем общее количество отсчетов для всей пачки
    samples_per_period = int(T_sl * f_s)
    total_samples = int(N_u * T_sl * f_s)
    
    tx_signal_full = np.zeros(total_samples, dtype=complex)
    
    # Расставляем импульсы в массиве
    for i in range(N_u):
        start_idx = i * samples_per_period
        end_idx = start_idx + len(tx_pulse)
        if end_idx <= total_samples:
            tx_signal_full[start_idx:end_idx] = tx_pulse
            
    t_full = np.arange(0, total_samples / f_s, 1/f_s)

    # ---------------------------------------------------------
    # ЧАСТЬ 1: ОСШ и Спектр
    # ---------------------------------------------------------
    snr_levels = [20, 0, -10] 
    
    # Сигнал для спектра (берем всю пачку с хорошим ОСШ)
    rx_for_spectrum = add_awgn(tx_signal_full, 20)
    
    plt.figure(figsize=(15, 15))
    plt.suptitle(f'Сигнал: Пачка из {N_u} имп. (τи={tau_u*1e6:.1f} мкс, T={T_sl*1e6:.1f} мкс)', fontsize=16)
    
    # 1.1 Временная область
    plt.subplot(3, 2, 1)
    plt.plot(t_full * 1e6, np.real(rx_for_spectrum), color='black')
    plt.title('Временная область (Реальная часть, ОСШ=20 дБ)')
    plt.xlabel('Время, мкс')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    
    # Масштабирование по оси X: показываем минимум 3 периода, но не больше всей пачки
    view_duration = max(3 * T_sl, min(N_u * T_sl, 5 * T_sl)) * 1e6 
    if N_u == 1: view_duration = tau_u * 2 * 1e6 # Если 1 импульс, зумим на него
    plt.xlim(0, view_duration)
    
    # 1.2 Спектральная область
    plt.subplot(3, 2, 2)
    spectrum = np.fft.fftshift(np.fft.fft(rx_for_spectrum))
    freqs = np.fft.fftshift(np.fft.fftfreq(total_samples, 1/f_s)) / 1e6
    
    plt.plot(freqs, 20 * np.log10(np.abs(spectrum) / np.max(np.abs(spectrum))))
    plt.title('Спектр пачки')
    plt.xlabel('Частота, МГц')
    plt.ylabel('Амплитуда, дБ')
    plt.grid(True)
    plt.xlim(f_if/1e6 - delta_f/1e6, f_if/1e6 + delta_f*2/1e6)
    
    # 1.3 Выход СФ для разных ОСШ
    colors = ['green', 'blue', 'red']
    plt.subplot(3, 1, 2)
    
    for i, snr in enumerate(snr_levels):
        rx_noisy = add_awgn(tx_signal_full, snr)
        
        # СФ настраиваем на ОДИНОЧНЫЙ импульс, чтобы видеть сжатие каждого импульса в пачке
        mf_out = matched_filter(rx_noisy, tx_pulse)
        mf_out_db = 20 * np.log10(np.abs(mf_out) / np.max(np.abs(mf_out)))
        
        t_mf = np.arange(0, len(mf_out)/f_s, 1/f_s) * 1e6 
        plt.plot(t_mf, mf_out_db, color=colors[i], label=f'ОСШ = {snr} дБ', alpha=0.8)
        
    plt.title('Выход СФ (Настроен на 1 импульс)')
    plt.xlabel('Время, мкс')
    plt.ylabel('Амплитуда, дБ')
    plt.grid(True)
    # Показываем несколько пиков на выходе фильтра
    plt.xlim(0, view_duration)
    plt.ylim(-40, 5)
    plt.legend()

    # ---------------------------------------------------------
    # ЧАСТЬ 2: Доплеровский сдвиг (Анализ для первого импульса)
    # ---------------------------------------------------------
    # Примечание: Здесь мы оставляем анализ одиночного импульса для наглядности физики эффекта,
    # так как на пачке картинка становится слишком сложной для базового графика.
    
    fd_1 = 0
    fd_2 = 1 / (4 * tau_u)  
    fd_3 = 1 / tau_u         
    fd_levels = [fd_1, fd_2, fd_3]
    fd_labels = [f'Fd = 0 Гц', 
                 f'Fd = {fd_2/1e3:.1f} кГц', 
                 f'Fd = {fd_3/1e3:.1f} кГц']

    plt.subplot(3, 1, 3)
    for i, fd in enumerate(fd_levels):
        # Генерируем один импульс со сдвигом
        rx_doppler_pulse, _ = generate_lfm_pulse(tau_u, f_if + fd, delta_f, f_s)
        
        # Создаем короткий буфер для этого демо (один период)
        rx_doppler_demo = np.zeros(samples_per_period, dtype=complex)
        rx_doppler_demo[:len(rx_doppler_pulse)] = rx_doppler_pulse
        
        mf_out_doppler = matched_filter(rx_doppler_demo, tx_pulse)
        mf_out_doppler_db = 20 * np.log10(np.abs(mf_out_doppler) / np.max(np.abs(mf_out_doppler)))
        
        t_mf_demo = np.arange(0, len(mf_out_doppler)/f_s, 1/f_s) * 1e6
        plt.plot(t_mf_demo, mf_out_doppler_db, color=colors[i], label=fd_labels[i], linewidth=1.5 if fd==0 else 1)
        
        if fd != 0:
            doppler_shift_time = -fd / (delta_f / tau_u)
            print(f"  [Доплер] Fd={fd/1e3:.1f} кГц -> сдвиг пика: {doppler_shift_time*1e6:.2f} мкс")

    plt.title('Влияние Доплера (на примере одного импульса)')
    plt.xlabel('Время, мкс')
    plt.ylabel('Амплитуда, дБ (нормир.)')
    plt.grid(True)
    plt.xlim(-tau_u*1.2*1e6, tau_u*1.2*1e6) 
    plt.ylim(-40, 5)
    plt.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# ==========================================
# ГЛАВНЫЙ ЦИКЛ
# ==========================================

def main():
    print("========================================")
    print("     МОДЕЛЬ ЛЧМ СИГНАЛА И ПАЧКИ")
    print("========================================")
    
    d_tau_u = 10e-6
    d_T_sl = 100e-6
    d_N_u = 1
    d_delta_f = 10e6
    d_f_c = 200e6
    d_f_if = 50e6
    d_f_s = 400e6

    while True:
        print("\n--- Ввод параметров (Enter - по умолчанию) ---")
        
        tau_u = get_user_input("Длительность импульса (tau_u)", d_tau_u, "мкс", 1e-6)
        T_sl = get_user_input("Период следования (T_sl)", d_T_sl, "мкс", 1e-6)
        
        # ВОТ НОВЫЙ ПАРАМЕТР
        val_nu = get_user_input("Кол-во импульсов в пачке (N_u)", d_N_u, "шт", 1.0)
        N_u = int(val_nu) # Преобразуем в целое число
        
        delta_f = get_user_input("Девиация частоты (delta_f)", d_delta_f, "МГц", 1e6)
        f_if = get_user_input("Промежуточная частота (f_if)", d_f_if, "МГц", 1e6)
        f_c = get_user_input("Несущая частота (f_c)", d_f_c, "МГц", 1e6)
        f_s = get_user_input("Частота дискретизации (f_s)", d_f_s, "МГц", 1e6)
        
        if T_sl <= tau_u:
            print(f"\n[ОШИБКА] T_sl должен быть больше tau_u!")
            continue

        print("\nЗапуск моделирования...")
        run_simulation(tau_u, T_sl, N_u, delta_f, f_c, f_if, f_s)
        
        answer = input("\nПродолжить с новыми параметрами? (y/n): ").lower().strip()
        if answer not in ['y', 'yes', 'д', 'да']:
            print("Завершение работы.")
            break

if __name__ == "__main__":
    main()