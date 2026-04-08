import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift
from scipy.signal import correlate
import warnings
warnings.filterwarnings('ignore')

# Настройка параметров отображения
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 100

# ==================== ОБЩИЕ ПАРАМЕТРЫ ====================
IF = 50e6           # Промежуточная частота, Гц
Fs = 200e6          # Частота дискретизации, Гц
Ts = 1 / Fs         # Период дискретизации, с
Ta = 50e-6          # Время анализа, с
c0 = 3e8            # Скорость света, м/с
lambda_c = 0.03     # Длина волны, м (для f=10 ГГц)

# Вектор времени
t = np.arange(0, Ta, Ts)
Nt = len(t)

# Вектор частот
f = np.arange(Nt) * Fs / Nt

# ==================== ФУНКЦИЯ НЕОПРЕДЕЛЕННОСТИ (ПОЛНАЯ) ====================
def ambiguity_function_full(U, t, Ts, tau_min, tau_max, N_tau, f_max_display):
    """
    Расчет полной функции неопределенности сигнала
    Возвращает матрицу ФН для отрицательных и положительных задержек и частот
    """
    # Вектор задержек (симметричный)
    tau_vec = np.linspace(tau_min, tau_max, N_tau)
    Nt = len(U)
    
    # Дополнительная длина для нулевого заполнения
    max_shift = int(np.ceil(max(abs(tau_min), abs(tau_max)) / Ts))
    N_pad = Nt + 2 * max_shift
    
    # Дополняем сигнал нулями
    U_padded = np.pad(U, (max_shift, max_shift), 'constant')
    
    # Инициализация матрицы ФН
    amf = np.zeros((N_tau, N_pad), dtype=complex)
    
    for i, tau in enumerate(tau_vec):
        shift_samples = int(np.round(tau / Ts))
        
        # Сдвиг сигнала
        if shift_samples >= 0:
            U_shifted = np.roll(U_padded, shift_samples)
            U_shifted[:shift_samples] = 0
        else:
            U_shifted = np.roll(U_padded, shift_samples)
            U_shifted[shift_samples:] = 0
        
        # Произведение
        product = U_padded * np.conj(U_shifted)
        
        # БПФ (без fftshift, чтобы получить полный спектр)
        amf[i, :] = fft(product)
    
    # Вектор частот (симметричный, от -Fs/2 до Fs/2)
    f_full = np.arange(N_pad) * Fs / N_pad
    f_full = f_full - f_full[N_pad//2]
    
    # Применяем fftshift к матрице ФН для симметричного отображения
    amf = fftshift(amf, axes=1)
    
    # Ограничиваем частотный диапазон
    idx_f = np.abs(f_full) <= f_max_display
    f_plot = f_full[idx_f] / 1e6  # в МГц
    amf_cut = amf[:, idx_f]
    
    # Нормировка
    amf_abs = np.abs(amf_cut)
    if np.max(amf_abs) > 0:
        amf_abs = amf_abs / np.max(amf_abs)
    
    return amf_abs, tau_vec, f_plot

# ==================== ЗАДАНИЕ 1: Импульсный сигнал (одиночный импульс) ====================
print("Задание 1: Одиночный прямоугольный импульс")

Tu1 = 0.25e-6       # Длительность импульса, с
Tpr1 = 1.5e-6       # Период следования, с
N_pt1 = 1           # Количество импульсов

# Модель сигнала
S_ampl1 = np.zeros(Nt, dtype=float)
for n in range(N_pt1):
    t_start = n * Tpr1
    t_end = t_start + Tu1
    mask = (t >= t_start) & (t < t_end)
    S_ampl1[mask] = 1.0

U1 = S_ampl1 * np.exp(1j * 2 * np.pi * IF * t)

# Спектр
S1 = fft(U1)

# АКФ
B1 = correlate(U1, U1, mode='full')
tau1 = np.linspace(-Ta, Ta, len(B1))

# Функция неопределенности
tau_min1 = -Tu1
tau_max1 = Tu1
N_tau1 = 200
f_max_display1 = 40e6  # ±40 МГц

amf1, tau1_af, f1_plot = ambiguity_function_full(U1, t, Ts, tau_min1, tau_max1, N_tau1, f_max_display1)

print(f"  Размерность ФН: {amf1.shape}")

# Построение графиков
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Задание 1: Одиночный прямоугольный импульс (τи = 0.25 мкс)', fontsize=14)

# Исходный сигнал
axes[0, 0].plot(t*1e6, np.real(U1), 'b', label='Re', linewidth=0.8)
axes[0, 0].plot(t*1e6, np.imag(U1), 'r', label='Im', linewidth=0.8)
axes[0, 0].set_xlabel('t, мкс')
axes[0, 0].set_ylabel('U(t), B')
axes[0, 0].set_title('Исходный сигнал')
axes[0, 0].legend()
axes[0, 0].grid(True)
axes[0, 0].set_xlim([0, 2])

# Амплитудный спектр
f_plot_fft = f - IF
idx_plot = np.abs(f_plot_fft) <= 40e6
axes[0, 1].plot(f_plot_fft[idx_plot]/1e6, np.abs(S1[idx_plot]), 'g', linewidth=0.8)
axes[0, 1].set_xlabel('f - f₀, МГц')
axes[0, 1].set_ylabel('|S(f)|, В/Гц')
axes[0, 1].set_title('Модуль амплитудного спектра')
axes[0, 1].grid(True)

# АКФ
tau_us = tau1 * 1e6
B1_norm = np.abs(B1) / np.max(np.abs(B1))
idx_akf = np.abs(tau_us) <= 2
axes[1, 0].plot(tau_us[idx_akf], B1_norm[idx_akf], 'm', linewidth=0.8)
axes[1, 0].set_xlabel('τ, мкс')
axes[1, 0].set_ylabel('|B(τ)|, отн.ед.')
axes[1, 0].set_title('Нормированная АКФ')
axes[1, 0].grid(True)
axes[1, 0].set_ylim([-0.1, 1.1])

# Функция неопределенности (сечения)
contour_levels = [0.1, 0.5, 0.707]
contour1 = axes[1, 1].contour(f1_plot, tau1_af*1e6, amf1, 
                               contour_levels, colors='k', linewidths=1.5)
axes[1, 1].clabel(contour1, inline=True, fontsize=8)
axes[1, 1].set_xlabel('f - f₀, МГц')
axes[1, 1].set_ylabel('τ, мкс')
axes[1, 1].set_title('Сечения ФН (уровни 0.1, 0.5, 0.707)')
axes[1, 1].grid(True)
axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
axes[1, 1].set_xlim([-40, 40])
axes[1, 1].set_ylim([-0.5, 0.5])

plt.tight_layout()
plt.savefig('task1_pulse.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  Разрешение по дальности: {c0 * Tu1 / 2:.2f} м")
print(f"  Разрешение по скорости: {(1/Tu1) * lambda_c / 2:.2f} м/с")
print()

# ==================== ЗАДАНИЕ 2: Сигнал с ЛЧМ ====================
print("Задание 2: Сигнал с ЛЧМ (одиночный импульс)")

Tu2 = 1e-6          # Длительность импульса, с
Tpr2 = 5e-6         # Период следования, с
N_pt2 = 1           # Количество импульсов
deltaF2 = 20e6      # Девиация частоты, Гц

# Модель сигнала с ЛЧМ
S_ampl2 = np.zeros(Nt, dtype=float)
for n in range(N_pt2):
    t_start = n * Tpr2
    t_end = t_start + Tu2
    mask = (t >= t_start) & (t < t_end)
    S_ampl2[mask] = 1.0

U2 = np.zeros(Nt, dtype=complex)
for n in range(N_pt2):
    t_start = n * Tpr2
    t_end = t_start + Tu2
    mask = (t >= t_start) & (t < t_end)
    t_local = t[mask] - t_start
    # Линейная частотная модуляция
    phase = 2 * np.pi * IF * t_local + 2 * np.pi * (deltaF2/(2*Tu2)) * t_local**2
    U2[mask] = S_ampl2[mask] * np.exp(1j * phase)

# Спектр
S2 = fft(U2)

# АКФ
B2 = correlate(U2, U2, mode='full')
tau2 = np.linspace(-Ta, Ta, len(B2))

# Функция неопределенности
tau_min2 = -Tu2
tau_max2 = Tu2
N_tau2 = 200
f_max_display2 = 25e6  # ±25 МГц

amf2, tau2_af, f2_plot = ambiguity_function_full(U2, t, Ts, tau_min2, tau_max2, N_tau2, f_max_display2)

print(f"  Размерность ФН: {amf2.shape}")

# Построение графиков
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Задание 2: Сигнал с ЛЧМ (τи = {Tu2*1e6:.2f} мкс, Δf = {deltaF2/1e6:.0f} МГц)', fontsize=14)

# Исходный сигнал (реальная часть)
axes[0, 0].plot(t*1e6, np.real(U2), 'b', linewidth=0.8)
axes[0, 0].set_xlabel('t, мкс')
axes[0, 0].set_ylabel('U(t), B')
axes[0, 0].set_title('Исходный сигнал (Re)')
axes[0, 0].grid(True)
axes[0, 0].set_xlim([0, 5])

# Амплитудный спектр
f_plot_fft = f - IF
idx_plot = np.abs(f_plot_fft) <= 25e6
axes[0, 1].plot(f_plot_fft[idx_plot]/1e6, np.abs(S2[idx_plot]), 'g', linewidth=0.8)
axes[0, 1].set_xlabel('f - f₀, МГц')
axes[0, 1].set_ylabel('|S(f)|, В/Гц')
axes[0, 1].set_title('Модуль амплитудного спектра')
axes[0, 1].grid(True)

# АКФ
tau_us = tau2 * 1e6
B2_norm = np.abs(B2) / np.max(np.abs(B2))
idx_akf = np.abs(tau_us) <= 2
axes[1, 0].plot(tau_us[idx_akf], B2_norm[idx_akf], 'm', linewidth=0.8)
axes[1, 0].set_xlabel('τ, мкс')
axes[1, 0].set_ylabel('|B(τ)|, отн.ед.')
axes[1, 0].set_title('Нормированная АКФ')
axes[1, 0].grid(True)

# Функция неопределенности (3D) - инвертированные цвета (синий=0дБ, красный=много дБ)
X, Y = np.meshgrid(f2_plot, tau2_af*1e6)
# Инвертируем цветовую карту: чем больше дБ, тем краснее
surf = axes[1, 1].pcolormesh(X, Y, 20*np.log10(amf2 + 1e-10), 
                              shading='auto', cmap='jet_r')  # jet_r - инвертированная цветовая карта
axes[1, 1].set_xlabel('f - f₀, МГц')
axes[1, 1].set_ylabel('τ, мкс')
axes[1, 1].set_title('ФН сигнала с ЛЧМ (дБ) - синий:0 дБ, красный:много дБ')
cbar = plt.colorbar(surf, ax=axes[1, 1], label='дБ')
axes[1, 1].axhline(y=0, color='w', linestyle='--', alpha=0.5)
axes[1, 1].axvline(x=0, color='w', linestyle='--', alpha=0.5)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task2_LFM.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  Разрешение по дальности (после сжатия): {c0 / (2 * deltaF2):.2f} м")
print(f"  Разрешение по скорости: {(1/Tu2) * lambda_c / 2:.2f} м/с")
print()

# ==================== ЗАДАНИЕ 3: Код Баркера ====================
print("Задание 3: Код Баркера")

code_barker = [1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1]  # Длина 13
Tu3 = 0.25e-6       # Длительность элементарного импульса, с
Tpr3 = Tu3 * len(code_barker)

# Формирование сигнала с кодом Баркера
S_ampl3 = np.zeros(Nt, dtype=float)
for i, code_val in enumerate(code_barker):
    t_start = i * Tu3
    t_end = t_start + Tu3
    mask = (t >= t_start) & (t < t_end)
    S_ampl3[mask] = code_val

U3 = S_ampl3 * np.exp(1j * 2 * np.pi * IF * t)

# Спектр
S3 = fft(U3)

# АКФ
B3 = correlate(U3, U3, mode='full')
tau3 = np.linspace(-Ta, Ta, len(B3))

# Функция неопределенности
tau_min3 = -10 * Tu3
tau_max3 = 10 * Tu3
N_tau3 = 300
f_max_display3 = 8e6  # ±8 МГц

amf3, tau3_af, f3_plot = ambiguity_function_full(U3, t, Ts, tau_min3, tau_max3, N_tau3, f_max_display3)

print(f"  Длина кода Баркера: {len(code_barker)}")
print(f"  Размерность ФН: {amf3.shape}")

# Построение графиков
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Задание 3: Код Баркера (длина {len(code_barker)}, τэ = {Tu3*1e6:.2f} мкс)', fontsize=14)

# Исходный сигнал (огибающая)
axes[0, 0].plot(t*1e6, S_ampl3, 'b', linewidth=0.8)
axes[0, 0].set_xlabel('t, мкс')
axes[0, 0].set_ylabel('A(t), отн.ед.')
axes[0, 0].set_title('Огибающая сигнала (код Баркера)')
axes[0, 0].grid(True)
axes[0, 0].set_xlim([0, Tpr3*1e6 + 1])

# Амплитудный спектр
f_plot_fft = f - IF
idx_plot = np.abs(f_plot_fft) <= 8e6
axes[0, 1].plot(f_plot_fft[idx_plot]/1e6, np.abs(S3[idx_plot]), 'g', linewidth=0.8)
axes[0, 1].set_xlabel('f - f₀, МГц')
axes[0, 1].set_ylabel('|S(f)|, В/Гц')
axes[0, 1].set_title('Модуль амплитудного спектра')
axes[0, 1].grid(True)

# АКФ
tau_us = tau3 * 1e6
B3_norm = np.abs(B3) / np.max(np.abs(B3))
idx_akf = np.abs(tau_us) <= 4
axes[1, 0].plot(tau_us[idx_akf], B3_norm[idx_akf], 'm', linewidth=0.8)
axes[1, 0].set_xlabel('τ, мкс')
axes[1, 0].set_ylabel('|B(τ)|, отн.ед.')
axes[1, 0].set_title('Нормированная АКФ кода Баркера')
axes[1, 0].grid(True)

# Функция неопределенности (сечения)
contour_levels = [0.1, 0.5, 0.707]
contour3 = axes[1, 1].contour(f3_plot, tau3_af*1e6, amf3, 
                               contour_levels, colors='k', linewidths=1.5)
axes[1, 1].clabel(contour3, inline=True, fontsize=8)
axes[1, 1].set_xlabel('f - f₀, МГц')
axes[1, 1].set_ylabel('τ, мкс')
axes[1, 1].set_title('Сечения ФН (уровни 0.1, 0.5, 0.707)')
axes[1, 1].grid(True)
axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('task3_Barker.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  Разрешение по дальности: {c0 * Tu3 / 2:.2f} м")
print(f"  Разрешение по скорости: {(1/(len(code_barker)*Tu3)) * lambda_c / 2:.2f} м/с")
print(f"  База сигнала: {len(code_barker)}")
print()

# ==================== ЗАДАНИЕ 4: М-последовательность ====================
print("Задание 4: М-последовательность")

code_m = [1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1]  # Длина 15
Tu4 = 0.25e-6       # Длительность элементарного импульса, с
Tpr4 = Tu4 * len(code_m)

# Формирование сигнала с М-последовательностью
S_ampl4 = np.zeros(Nt, dtype=float)
for i, code_val in enumerate(code_m):
    t_start = i * Tu4
    t_end = t_start + Tu4
    mask = (t >= t_start) & (t < t_end)
    S_ampl4[mask] = code_val

U4 = S_ampl4 * np.exp(1j * 2 * np.pi * IF * t)

# Спектр
S4 = fft(U4)

# АКФ
B4 = correlate(U4, U4, mode='full')
tau4 = np.linspace(-Ta, Ta, len(B4))

# Функция неопределенности
tau_min4 = -10 * Tu4
tau_max4 = 10 * Tu4
N_tau4 = 300
f_max_display4 = 8e6  # ±8 МГц

amf4, tau4_af, f4_plot = ambiguity_function_full(U4, t, Ts, tau_min4, tau_max4, N_tau4, f_max_display4)

print(f"  Длина М-последовательности: {len(code_m)}")
print(f"  Размерность ФН: {amf4.shape}")

# Построение графиков
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Задание 4: М-последовательность (длина {len(code_m)}, τэ = {Tu4*1e6:.2f} мкс)', fontsize=14)

# Исходный сигнал (огибающая)
axes[0, 0].plot(t*1e6, S_ampl4, 'b', linewidth=0.8)
axes[0, 0].set_xlabel('t, мкс')
axes[0, 0].set_ylabel('A(t), отн.ед.')
axes[0, 0].set_title('Огибающая сигнала (М-последовательность)')
axes[0, 0].grid(True)
axes[0, 0].set_xlim([0, Tpr4*1e6 + 1])

# Амплитудный спектр
f_plot_fft = f - IF
idx_plot = np.abs(f_plot_fft) <= 8e6
axes[0, 1].plot(f_plot_fft[idx_plot]/1e6, np.abs(S4[idx_plot]), 'g', linewidth=0.8)
axes[0, 1].set_xlabel('f - f₀, МГц')
axes[0, 1].set_ylabel('|S(f)|, В/Гц')
axes[0, 1].set_title('Модуль амплитудного спектра')
axes[0, 1].grid(True)

# АКФ
tau_us = tau4 * 1e6
B4_norm = np.abs(B4) / np.max(np.abs(B4))
idx_akf = np.abs(tau_us) <= 4
axes[1, 0].plot(tau_us[idx_akf], B4_norm[idx_akf], 'm', linewidth=0.8)
axes[1, 0].set_xlabel('τ, мкс')
axes[1, 0].set_ylabel('|B(τ)|, отн.ед.')
axes[1, 0].set_title('Нормированная АКФ М-последовательности')
axes[1, 0].grid(True)

# Функция неопределенности (сечения)
contour_levels = [0.1, 0.5, 0.707]
contour4 = axes[1, 1].contour(f4_plot, tau4_af*1e6, amf4, 
                               contour_levels, colors='k', linewidths=1.5)
axes[1, 1].clabel(contour4, inline=True, fontsize=8)
axes[1, 1].set_xlabel('f - f₀, МГц')
axes[1, 1].set_ylabel('τ, мкс')
axes[1, 1].set_title('Сечения ФН (уровни 0.1, 0.5, 0.707)')
axes[1, 1].grid(True)
axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('task4_MPseudo.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  Разрешение по дальности: {c0 * Tu4 / 2:.2f} м")
print(f"  Разрешение по скорости: {(1/(len(code_m)*Tu4)) * lambda_c / 2:.2f} м/с")
print(f"  База сигнала: {len(code_m)}")
print()

# ==================== ЗАДАНИЕ 5: Пачка импульсов ====================
print("Задание 5: Пачка прямоугольных импульсов")

Tu5 = 0.25e-6       # Длительность импульса, с
Tpr5 = 1.5e-6       # Период следования, с
N_pt5 = 4           # Количество импульсов в пачке

# Модель пачки импульсов
S_ampl5 = np.zeros(Nt, dtype=float)
for n in range(N_pt5):
    t_start = n * Tpr5
    t_end = t_start + Tu5
    mask = (t >= t_start) & (t < t_end)
    S_ampl5[mask] = 1.0

U5 = S_ampl5 * np.exp(1j * 2 * np.pi * IF * t)

# Спектр
S5 = fft(U5)

# АКФ
B5 = correlate(U5, U5, mode='full')
tau5 = np.linspace(-Ta, Ta, len(B5))

# Функция неопределенности
tau_min5 = -8 * Tu5
tau_max5 = 8 * Tu5
N_tau5 = 300
f_max_display5 = 40e6  # ±40 МГц

amf5, tau5_af, f5_plot = ambiguity_function_full(U5, t, Ts, tau_min5, tau_max5, N_tau5, f_max_display5)

print(f"  Количество импульсов в пачке: {N_pt5}")
print(f"  Период следования: {Tpr5*1e6:.2f} мкс")
print(f"  Размерность ФН: {amf5.shape}")

# Построение графиков
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Задание 5: Пачка импульсов (N = {N_pt5}, Tпр = {Tpr5*1e6:.2f} мкс, τи = {Tu5*1e6:.2f} мкс)', fontsize=14)

# Исходный сигнал
axes[0, 0].plot(t*1e6, S_ampl5, 'b', linewidth=0.8)
axes[0, 0].set_xlabel('t, мкс')
axes[0, 0].set_ylabel('A(t), отн.ед.')
axes[0, 0].set_title('Огибающая пачки импульсов')
axes[0, 0].grid(True)
axes[0, 0].set_xlim([0, N_pt5 * Tpr5 * 1e6 + 1])

# Амплитудный спектр
f_plot_fft = f - IF
idx_plot = np.abs(f_plot_fft) <= 40e6
axes[0, 1].plot(f_plot_fft[idx_plot]/1e6, np.abs(S5[idx_plot]), 'g', linewidth=0.8)
axes[0, 1].set_xlabel('f - f₀, МГц')
axes[0, 1].set_ylabel('|S(f)|, В/Гц')
axes[0, 1].set_title('Модуль амплитудного спектра')
axes[0, 1].grid(True)

# АКФ
tau_us = tau5 * 1e6
B5_norm = np.abs(B5) / np.max(np.abs(B5))
idx_akf = np.abs(tau_us) <= 10
axes[1, 0].plot(tau_us[idx_akf], B5_norm[idx_akf], 'm', linewidth=0.8)
axes[1, 0].set_xlabel('τ, мкс')
axes[1, 0].set_ylabel('|B(τ)|, отн.ед.')
axes[1, 0].set_title('Нормированная АКФ пачки импульсов')
axes[1, 0].grid(True)

# Функция неопределенности (сечения)
contour_levels = [0.1, 0.5, 0.707]
contour5 = axes[1, 1].contour(f5_plot, tau5_af*1e6, amf5, 
                               contour_levels, colors='k', linewidths=1.5)
axes[1, 1].clabel(contour5, inline=True, fontsize=8)
axes[1, 1].set_xlabel('f - f₀, МГц')
axes[1, 1].set_ylabel('τ, мкс')
axes[1, 1].set_title('Сечения ФН (уровни 0.1, 0.5, 0.707)')
axes[1, 1].grid(True)
axes[1, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('task5_Burst.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  Разрешение по дальности (одиночный импульс): {c0 * Tu5 / 2:.2f} м")
print(f"  Разрешение по скорости (пачка): {(1/(N_pt5 * Tpr5)) * lambda_c / 2:.2f} м/с")
print()

# ==================== ФН в координатах (R, V) ====================
print("\n ФН в координатах дальность-скорость для пачки импульсов")

# Параметры сигнала
Tu5 = 0.25e-6       # Длительность импульса, с
Tpr5 = 1.5e-6       # Период следования, с
N_pt5 = 4           # Количество импульсов в пачке
f0 = 10e9           # Рабочая частота РЛС, Гц
c0 = 3e8            # Скорость света

# Физические пределы
R_max_unambiguous = c0 * Tpr5 / 2  # Период неоднозначности по дальности
delta_R = c0 * Tu5 / 2              # Разрешение по дальности
delta_V = c0 / (2 * f0 * N_pt5 * Tpr5)  # Разрешение по скорости

print(f"\nФизические параметры сигнала:")
print(f"  Длительность импульса: {Tu5*1e6:.2f} мкс")
print(f"  Период повторения: {Tpr5*1e6:.2f} мкс")
print(f"  Количество импульсов: {N_pt5}")
print(f"  Разрешение по дальности: {delta_R:.1f} м")
print(f"  Разрешение по скорости: {delta_V:.2f} м/с")
print(f"  Период неоднозначности по дальности: {R_max_unambiguous:.1f} м")

# Выбираем разумные пределы для отображения
# Обычно достаточно показать ±3-5 периодов повторения по дальности
R_limit = min(3 * R_max_unambiguous, 1000)  # Не более 1000 м для наглядности
# По скорости - ± несколько доплеровских периодов
V_limit = 3 * c0 / (2 * f0 * Tpr5)  # 3 периода неоднозначности по скорости

print(f"\nВыбранные пределы для графика:")
print(f"  Дальность: ±{R_limit:.1f} м")
print(f"  Скорость: ±{V_limit:.1f} м/с")

# ПРАВИЛЬНЫЙ пересчет в координаты дальности и скорости
R_tau = tau5_af * c0 / 2
V_f = f5_plot * 1e6 * c0 / (2 * f0)

# Обрезаем до разумных пределов
mask_R = np.abs(R_tau) <= R_limit
mask_V = np.abs(V_f) <= V_limit

R_tau_cut = R_tau[mask_R]
V_f_cut = V_f[mask_V]
amf5_cut = amf5[np.ix_(mask_R, mask_V)]

print(f"  Размерность после обрезания: {amf5_cut.shape}")

# Создаем наглядный график
fig, ax = plt.subplots(figsize=(12, 10))

# Создаем сетку для contourf
X_grid, Y_grid = np.meshgrid(V_f_cut, R_tau_cut)

# Основной график - цветная заливка
levels = np.linspace(0, 1, 50)
cf = ax.contourf(X_grid, Y_grid, amf5_cut, levels=levels, 
                  cmap='viridis', alpha=0.9)

# Контурные линии на заданных уровнях
contour_levels_rv = [0.1, 0.5, 0.707]
contour_RV = ax.contour(X_grid, Y_grid, amf5_cut, contour_levels_rv, 
                         colors='red', linewidths=2.5, linestyles='solid')
ax.clabel(contour_RV, inline=True, fontsize=11, fmt='%.1f')

# Подписи уровней на цветовой шкале
cbar = plt.colorbar(cf, ax=ax, label='|χ(τ,f)|, отн.ед.')
cbar.set_ticks([0, 0.1, 0.5, 0.707, 1])
cbar.set_ticklabels(['0', '0.1', '0.5', '0.707', '1'])

# Настройка осей
ax.set_xlabel(f'Скорость V, м/с\n(разрешение {delta_V:.1f} м/с)', fontsize=12)
ax.set_ylabel(f'Дальность R, м\n(разрешение {delta_R:.1f} м, период неоднозначности {R_max_unambiguous:.0f} м)', fontsize=12)
ax.set_title('Функция неопределенности пачки импульсов\nв координатах (дальность, скорость)', fontsize=14)

# Осевые линии
ax.axhline(y=0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)
ax.axvline(x=0, color='white', linestyle='--', alpha=0.7, linewidth=1.5)

# Установка пределов
ax.set_xlim([-V_limit, V_limit])
ax.set_ylim([-R_limit, R_limit])

# Добавление сетки
ax.grid(True, alpha=0.3, linestyle='--')

# Добавление подписей основных лепестков
ax.text(0, 0, 'Главный\nлепесток', ha='center', va='center', 
        fontsize=10, color='white', fontweight='bold')

# Аннотация с параметрами сигнала
param_text = f'Параметры сигнала:\n'
param_text += f'τи = {Tu5*1e6:.2f} мкс\n'
param_text += f'Tпр = {Tpr5*1e6:.2f} мкс\n'
param_text += f'N = {N_pt5}\n\n'
param_text += f'ΔR = {delta_R:.1f} м\n'
param_text += f'ΔV = {delta_V:.2f} м/с\n'
param_text += f'R_max = {R_max_unambiguous:.0f} м'

ax.annotate(param_text, xy=(0.02, 0.98), xycoords='axes fraction',
            fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('task5_ambiguity_RV.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nДополнительный график сохранен в task5_ambiguity_RV.png")