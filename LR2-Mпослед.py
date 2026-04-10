import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftshift, ifft

def main():
    # ==========================================
    # 1. ОБЩИЕ ПАРАМЕТРЫ (из методички)
    # ==========================================
    T_rep = 100e-6       # Период повторения (100 мкс)
    f0 = 200e6           # Несущая частота (200 МГц) - для справки
    f_IF = 50e6          # Промежуточная частота (50 МГц)
    Fs = 400e6           # Частота дискретизации (400 МГц)
    
    # Шаг дискретизации
    Ts = 1 / Fs
    
   
    t = np.arange(0, T_rep, Ts)
    N = len(t)
    
    print(f"--- Инициализация моделирования ---")
    print(f"Частота дискретизации: {Fs/1e6} МГц")
    print(f"Период повторения: {T_rep*1e6} мкс")
    print(f"Всего отсчетов на период: {N}")

    # ==========================================
    # 2. ВВОД ДАННЫХ ПОЛЬЗОВАТЕЛЕМ
    # ==========================================
    try:
        tu_input = input("Введите длительность импульса в мкс (например, 1): ")
        Tu = float(tu_input) * 1e-6
        
        code_str = input("Введите код последовательности через запятую (например, 1, -1, 1, -1):\n")
        # Преобразуем строку в список чисел
        code = [int(x.strip()) for x in code_str.split(',') if x.strip().isdigit() or x.strip() == '-1']
        code = np.array(code)
        
        if len(code) == 0:
            raise ValueError("Код не был введен корректно.")
            
    except ValueError as e:
        print(f"Ошибка ввода: {e}")
        # Используем значения по умолчанию (Таблица 3, вар 4)
        print("Используются значения по умолчанию: Tu = 1 мкс, Код из варианта 4.")
        Tu = 1e-6
        code = np.array([1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, 1, 1, -1])

    print(f"Длительность импульса: {Tu*1e6} мкс")
    print(f"Длина кода: {len(code)} элементов")

    # ==========================================
    # 3. ГЕНЕРАЦИЯ ЗОНДИРУЮЩЕГО СИГНАЛА (ФКМ на ПЧ)
    # ==========================================
    # Длительность одного чипа
    T_chip = Tu / len(code)
    
    # Создаем массив чипов, растянутых на время Tu
    # Нам нужно создать огибающую кода, дискретизированную с частотой Fs
    samples_per_chip = int(T_chip / Ts)
    if samples_per_chip < 1:
        print("Внимание: Длительность чипа меньше периода дискретизации! Код будет утерян.")
        samples_per_chip = 1

    # Формируем массив кода, повторяя каждый элемент
    baseband_code = np.repeat(code, samples_per_chip)
    
    # Обрезаем или дополняем, чтобы точно соответствовать Tu
    total_samples_pulse = int(Tu / Ts)
    baseband_code = baseband_code[:total_samples_pulse]
    if len(baseband_code) < total_samples_pulse:
        baseband_code = np.pad(baseband_code, (0, total_samples_pulse - len(baseband_code)))
        
    # Создаем полный вектор сигнала (заполняем нулями на периоде повторения)
    # Сигнал начинается с небольшой задержкой (например, 10 мкс), чтобы видеть фронт
    delay_samples = int(10e-6 / Ts) 
    signal_IF_complex = np.zeros(N, dtype=complex)
    
    # Заполняем комплексный сигнал на ПЧ: Code(t) * exp(j * 2 * pi * f_IF * t)
    # Учитываем задержку
    t_pulse = t[delay_samples : delay_samples + total_samples_pulse]
    
    # Комплексная огибающая с несущей частотой f_IF
    carrier_part = np.exp(1j * 2 * np.pi * f_IF * t_pulse)
    signal_IF_complex[delay_samples : delay_samples + total_samples_pulse] = baseband_code * carrier_part

    # ==========================================
    # 4. ФУНКЦИЯ СОГЛАСОВАННОЙ ФИЛЬТРАЦИИ
    # ==========================================
    def matched_filter(received_signal, reference_signal):
        """
        Выполняет свертку принятого сигнала с комплексно-сопряженным 
        обращенным во времени опорным сигналом.
        Используем FFT для ускорения.
        """
        # Опорный сигнал (эталон) - комплексно сопряженный и перевернутый
        h = np.conj(reference_signal[::-1])
        
        # Длина свертки
        n_conv = len(received_signal) + len(h) - 1
        
        # Берем БПФ
        S = fft(received_signal, n=n_conv)
        H = fft(h, n=n_conv)
        
        # Спектр произведения и обратное БПФ
        Y = S * H
        y = ifft(Y)
        
        return y

    # Эталонный сигнал для СФ (импульс без задержки, только сам код)
    ref_signal = np.zeros(N, dtype=complex)
    ref_signal[:total_samples_pulse] = baseband_code * np.exp(1j * 2 * np.pi * f_IF * t[:total_samples_pulse])

    # ==========================================
    # 5. ИССЛЕДОВАНИЕ 1: ВЛИЯНИЕ ШУМА (SNR)
    # ==========================================
    print("\n--- Моделирование влияния шума (SNR) ---")
    
    # Характерные значения SNR (дБ)
    snr_values_db = [-5, 10, 20] 
    # Задержка приема (цель неподвижна, просто возвращается с задержкой)
    # Для упрощения графиков считаем, что сигнал уже пришел и находится в начале массива
    # (используем signal_IF_complex как принятый с некоторой задержкой)
    
    rx_static = signal_IF_complex.copy()
    
    # Расчет мощности сигнала для генерации шума
    signal_power = np.mean(np.abs(rx_static)**2)

    fig_snr, axes_snr = plt.subplots(len(snr_values_db), 3, figsize=(15, 12))
    fig_snr.suptitle(f'Влияние шума на отраженный сигнал (Код: {len(code)} эл., Tu={Tu*1e6} мкс)', fontsize=14)

    for i, snr_db in enumerate(snr_values_db):
        # Генерация шума
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        # Комплексный шум (действительная и мнимая части)
        noise = (np.random.normal(0, 1, N) + 1j * np.random.normal(0, 1, N)) * np.sqrt(noise_power / 2)
        
        rx_noisy = rx_static + noise
        
        # --- Графики ---
        
        # 1. Временная область (действительная часть - то, что увидел бы осциллограф)
        axes_snr[i, 0].plot(t * 1e6, np.real(rx_noisy))
        axes_snr[i, 0].set_title(f'Сигнал во времени (SNR = {snr_db} дБ)')
        axes_snr[i, 0].set_xlabel('Время, мкс')
        axes_snr[i, 0].set_ylabel('Амплитуда (В)')
        axes_snr[i, 0].grid(True)
        axes_snr[i, 0].set_xlim(0, 20) # Масштабируем к началу, где импульс
        
        # 2. Спектральная область
        spectrum = fftshift(fft(rx_noisy))
        freq_axis = fftshift(np.fft.fftfreq(N, Ts))
        
        axes_snr[i, 1].plot(freq_axis / 1e6, 20 * np.log10(np.abs(spectrum) + 1e-9))
        axes_snr[i, 1].set_title(f'Спектр сигнала (SNR = {snr_db} дБ)')
        axes_snr[i, 1].set_xlabel('Частота, МГц')
        axes_snr[i, 1].set_ylabel('Амплитуда, дБ')
        axes_snr[i, 1].grid(True)
        axes_snr[i, 1].set_xlim(0, 100) # Вокруг ПЧ (50 МГц)
        axes_snr[i, 1].set_ylim(-40, np.max(20 * np.log10(np.abs(spectrum))) + 10)
        
        # 3. Выход Согласованного Фильтра
        mf_out = matched_filter(rx_noisy, ref_signal)
        mf_out_mag = np.abs(mf_out)
        t_mf = np.arange(len(mf_out)) * Ts * 1e6
        
        axes_snr[i, 2].plot(t_mf, mf_out_mag)
        axes_snr[i, 2].set_title(f'Отклик СФ (SNR = {snr_db} дБ)')
        axes_snr[i, 2].set_xlabel('Время, мкс')
        axes_snr[i, 2].set_ylabel('Амплитуда (о.е.)')
        axes_snr[i, 2].grid(True)
        # Показываем область вокруг пика
        peak_idx = np.argmax(mf_out_mag)
        center_us = t_mf[peak_idx]
        axes_snr[i, 2].set_xlim(center_us - 2, center_us + 2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('task1_snr_analysis.png')
    plt.show()

    # ==========================================
    # 6. ИССЛЕДОВАНИЕ 2: ЭФФЕКТ ДОПЛЕРА
    # ==========================================
    print("\n--- Моделирование эффекта Доплера ---")
    
    # Характерные значения доплеровского сдвига
    # 0 Гц (статика), малый сдвиг, значительный сдвиг (сравнимый с 1/Tu)
    # При Tu = 1 мкс, полоса ~1 МГц. Сдвиг 500 кГц даст сильные искажения.
    fd_values = [0, 100e3, 500e3] 
    
    fig_dop, axes_dop = plt.subplots(len(fd_values), 1, figsize=(10, 12))
    fig_dop.suptitle('Влияние доплеровского сдвига на сжатый импульс (Без шума)', fontsize=14)

    for i, fd in enumerate(fd_values):
        # Моделирование Доплера: умножение на exp(j * 2 * pi * fd * t)
        # Это эквивалентно сдвигу частоты принимаемого сигнала
        doppler_phase = np.exp(1j * 2 * np.pi * fd * t)
        rx_doppler = rx_static * doppler_phase
        
        # Согласованная фильтрация
        mf_out_dop = matched_filter(rx_doppler, ref_signal)
        mf_out_mag_dop = np.abs(mf_out_dop)
        t_mf_dop = np.arange(len(mf_out_dop)) * Ts * 1e6
        
        # График
        axes_dop[i].plot(t_mf_dop, mf_out_mag_dop, label=f'Fd = {fd/1e3} кГц', color='red')
        axes_dop[i].set_title(f'Отклик СФ при доплеровском сдвиге Fd = {fd/1e3} кГц')
        axes_dop[i].set_xlabel('Время, мкс')
        axes_dop[i].set_ylabel('Амплитуда (о.е.)')
        axes_dop[i].grid(True)
        
        # Масштабирование к пику для оценки искажений
        peak_idx = np.argmax(mf_out_mag_dop)
        center_us = t_mf_dop[peak_idx]
        axes_dop[i].set_xlim(center_us - 3, center_us + 3)
        axes_dop[i].legend()
        
        # Вывод уровня потерь в консоль
        peak_val = mf_out_mag_dop[peak_idx]
        # Нормируем относительно идеального случая (fd=0, если он есть в списке, или относительно энергии)
        # Для простоты сравним с максимумом текущего графика
        print(f"Fd = {fd/1000} кГц: Пик сжатия = {peak_val:.2f} (отн. ед)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('task2_doppler_analysis.png')
    plt.show()

if __name__ == "__main__":
    main()