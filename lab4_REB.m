%% =========================================================================
%%  ЛАБОРАТОРНАЯ РАБОТА №4
%%  "Исследование отражающей возможности объектов различной формы"
%%  Дисциплина: Теоретические основы радиоэлектронной борьбы
%%  Среда: MATLAB R2025b
%%
%%  Формулы по методическим указаниям:
%%    (2)  - ЭПР сферы, рэлеевская область (a << lambda)
%%    (5)  - ЭПР диска, коротковолновый режим (a >> lambda)
%%    (6)  - ЭПР диска, длинноволновый режим (a << lambda)
%%    (7)  - ЭПР прямоугольной пластины (a, b >> lambda)
%%    (8)  - ЭПР конуса
%%    (9)  - ЭПР двугранного уголка, плоскость ⊥ ребру
%%    (10) - ЭПР двугранного уголка, ортогональная плоскость
%%    (11) - Максимальные ЭПР трёхгранных уголков
%%    (12) - ЭПР линзы Люнеберга (полусфера)
%%    (13) - ЭПР линзы Люнеберга с кольцом
%%    (14) - ЭПР решётки Ван-Атта
%% =========================================================================

clear; close all; clc;

%% --- Белая тема для всех фигур (отключение тёмной темы MATLAB) ---
set(0, 'DefaultFigureColor',      'w');          % фон фигуры — белый
set(0, 'DefaultAxesColor',        'w');          % фон осей — белый
set(0, 'DefaultAxesXColor',       'k');          % ось X — чёрная
set(0, 'DefaultAxesYColor',       'k');          % ось Y — чёрная
set(0, 'DefaultAxesZColor',       'k');          % ось Z — чёрная
set(0, 'DefaultTextColor',        'k');          % текст — чёрный
set(0, 'DefaultAxesFontName',     'Times New Roman');  % шрифт
set(0, 'DefaultAxesFontSize',     11);           % размер шрифта осей
set(0, 'DefaultTextFontName',     'Times New Roman');
set(0, 'DefaultTextFontSize',     12);

%% =========================================================================
%%  БЛОК 0 — ПАРАМЕТРЫ ВАРИАНТА
%% =========================================================================

variant = 1;                 % Номер бригады (варианта)

% --- Частотные параметры ---
f       = 2e9;               % Несущая частота, Гц
c       = 3e8;               % Скорость света, м/с
lambda  = c / f;             % Длина волны, м
k       = 2 * pi / lambda;   % Волновое число, рад/м

% --- Геометрические параметры ---
a       = 10e-3;             % Размер a, м
b       = 21e-3;             % Размер b, м
l_param = 1.5;               % Размер l, м

% --- Вывод информации о варианте ---
fprintf('==========================================================\n');
fprintf('  Лабораторная работа №4 — Вариант %d\n', variant);
fprintf('==========================================================\n');
fprintf('  Несущая частота      f = %.1f ГГц\n', f/1e9);
fprintf('  Длина волны          lambda = %.4f м (%.1f мм)\n', lambda, lambda*1e3);
fprintf('  Волновое число       k = %.2f рад/м\n', k);
fprintf('  Параметр a           = %.1f мм\n', a*1e3);
fprintf('  Параметр b           = %.1f мм\n', b*1e3);
fprintf('  Параметр l           = %.2f м\n', l_param);
fprintf('  Электрический размер ka = %.3f\n', k*a);
fprintf('==========================================================\n\n');

%% =========================================================================
%%  ЗАДАНИЕ 1 — ЭПР МЕТАЛЛИЧЕСКОЙ СФЕРЫ (ШАРА) В РЭЛЕЕВСКОЙ ОБЛАСТИ
%%
%%  Условие: a << lambda
%%  Диапазон: a от 0.001*lambda до 0.1*lambda (>= 100 точек)
%%  Формула (2): sigma = 144 * pi^5 * a^6 / lambda^4
%%
%%  В рэлеевской области ЭПР сферы очень мала и резко растёт с увеличением
%%  радиуса (пропорционально a^6).
%% =========================================================================

fprintf('--- Задание 1: ЭПР сферы (Рэлеевская область) ---\n');

% Диапазон размеров сферы
N_pts    = 500;                           % Количество точек (>> 100)
a_sphere = linspace(0.001*lambda, 0.1*lambda, N_pts);  % Размеры a, м

% Формула (2): ЭПР сферы в рэлеевской области
sigma_sphere = 144 * pi^5 * a_sphere.^6 / lambda^4;    % м^2

% Перевод в дБм^2 (децибелы относительно 1 м^2)
sigma_sphere_dB = 10 * log10(sigma_sphere);

% Построение графика
figure('Name', 'Задание 1 — ЭПР сферы (Рэлеев)', 'NumberTitle', 'off');

subplot(2,1,1);
plot(a_sphere/lambda, sigma_sphere, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('a / \lambda');
ylabel('\sigma, м^2');
title(['ЭПР сферы: \sigma = 144\pi^5 a^6 / \lambda^4,  \lambda = ', ...
       num2str(lambda*1e3, '%.1f'), ' мм']);
xlim([0.001 0.1]);

subplot(2,1,2);
semilogy(a_sphere/lambda, sigma_sphere, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('a / \lambda');
ylabel('\sigma, м^2');
title('ЭПР сферы (логарифмический масштаб)');
xlim([0.001 0.1]);

% Вывод контрольных значений
fprintf('  a = 0.001*lambda: sigma = %.3e м^2 (%.1f дБм^2)\n', ...
    sigma_sphere(1), sigma_sphere_dB(1));
fprintf('  a = 0.01*lambda:  sigma = %.3e м^2 (%.1f дБм^2)\n', ...
    sigma_sphere(round(N_pts/10)), sigma_sphere_dB(round(N_pts/10)));
fprintf('  a = 0.1*lambda:   sigma = %.3e м^2 (%.1f дБм^2)\n\n', ...
    sigma_sphere(end), sigma_sphere_dB(end));


%% =========================================================================
%%  ЗАДАНИЕ 2 — МОНОСТАТИЧЕСКАЯ ЭПР МЕТАЛЛИЧЕСКОГО ДИСКА
%%
%%  Условие: a << lambda (длинноволновая область)
%%  Формула (6): sigma(theta) = sigma_m * (8*ka / (3*pi))^2 * cos^4(theta)
%%  Формула (4): sigma_m = 4*pi^3 * a^4 / lambda^2
%%
%%  Диск ориентирован нормально к направлению падения волны.
%%  theta = 0 — нормальное падение (максимум ЭПР).
%% =========================================================================

fprintf('--- Задание 2: ЭПР диска (длинноволновая область) ---\n');

% Максимальная ЭПР диска — формула (4)
sigma_m_disk = 4 * pi^3 * a^4 / lambda^2;  % м^2

% Коэффициент для формулы (6)
C_rayleigh = (8 * k * a / (3 * pi))^2;

fprintf('  Максимальная ЭПР диска (формула 4): sigma_m = %.3e м^2\n', sigma_m_disk);
fprintf('  Коэффициент (8ka/3pi)^2 = %.4f\n', C_rayleigh);

% Углы падения волны
theta_disk = linspace(0, pi/2, 500);  % от 0 до 90 градусов

% Формула (6): моностатическая ЭПР диска (a << lambda)
sigma_disk = sigma_m_disk * C_rayleigh * cos(theta_disk).^4;  % м^2

% Перевод в дБ
sigma_disk_dB = 10 * log10(sigma_disk);
sigma_disk_dB(sigma_disk_dB == -Inf) = -60;  % ограничение для графика

% Построение графика
figure('Name', 'Задание 2 — ЭПР диска', 'NumberTitle', 'off');

subplot(2,1,1);
plot(theta_disk * 180/pi, sigma_disk, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('\theta, градусы');
ylabel('\sigma, м^2');
title(['ЭПР диска (a << \lambda): a = ', num2str(a*1e3, '%.0f'), ...
       ' мм, \lambda = ', num2str(lambda*1e3, '%.1f'), ' мм'], 'Interpreter', 'tex');
xlim([0 90]);

subplot(2,1,2);
plot(theta_disk * 180/pi, sigma_disk_dB, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('\theta, градусы');
ylabel('\sigma, дБм^2');
title('ЭПР диска в децибелах');
xlim([0 90]);
ylim([-60 max(sigma_disk_dB)+5]);

fprintf('  sigma(0)   = %.3e м^2 (%.1f дБм^2)\n', sigma_disk(1), sigma_disk_dB(1));
fprintf('  sigma(45)  = %.3e м^2 (%.1f дБм^2)\n', ...
    sigma_disk(round(end/4)), sigma_disk_dB(round(end/4)));
fprintf('  sigma(90)  = %.3e м^2\n\n', sigma_disk(end));


%% =========================================================================
%%  ЗАДАНИЕ 3 — ЭПР ПРЯМОУГОЛЬНОЙ ПЛАСТИНЫ
%%
%%  Условие: a, b >> lambda
%%  Размеры: a*100 и b*100 (домножить на 100)
%%  Формула (7):
%%    sigma(theta,phi) = sigma_m * |cos(theta) *
%%        sinc(2ka*sin(theta)*cos(phi)) *
%%        sinc(2kb*sin(theta)*sin(phi))|^2
%%    sigma_m = 64*pi*a^2*b^2 / lambda^2
%%
%%  Необходимо:
%%    - Построить 3D-изображение sigma(theta, phi)
%%    - Построить сечения в плоскостях xz (phi=0) и yz (phi=pi/2)
%% =========================================================================

fprintf('--- Задание 3: ЭПР прямоугольной пластины ---\n');

% Размеры пластины (домножены на 100)
a_plate = a * 100;   % 1.0 м
b_plate = b * 100;   % 2.1 м

fprintf('  Размеры пластины: %.2f м x %.2f м\n', a_plate, b_plate);
fprintf('  a_plate / lambda = %.1f,  b_plate / lambda = %.1f\n', ...
    a_plate/lambda, b_plate/lambda);

% Максимальная ЭПР пластины
sigma_m_plate = 64 * pi * a_plate^2 * b_plate^2 / lambda^2;
fprintf('  sigma_m = %.3e м^2 (%.1f дБм^2)\n\n', sigma_m_plate, ...
    10*log10(sigma_m_plate));

% Сетка углов
N_theta = 200;
N_phi   = 200;
theta_plate = linspace(0.001, pi/2, N_theta);  % избегаем theta=0 (деление на ноль)
phi_plate   = linspace(-pi, pi, N_phi);

[THETA, PHI] = meshgrid(theta_plate, phi_plate);

% Вспомогательные аргументы синуса
arg_a = 2 * k * a_plate * sin(THETA) .* cos(PHI);
arg_b = 2 * k * b_plate * sin(THETA) .* sin(PHI);

% Безопасное вычисление sinc(x) = sin(x)/x
sinc_a = ones(size(arg_a));
sinc_b = ones(size(arg_b));
mask_a = abs(arg_a) > 1e-10;
mask_b = abs(arg_b) > 1e-10;
sinc_a(mask_a) = sin(arg_a(mask_a)) ./ arg_a(mask_a);
sinc_b(mask_b) = sin(arg_b(mask_b)) ./ arg_b(mask_b);

% Формула (7): ЭПР прямоугольной пластины
sigma_plate = sigma_m_plate * (cos(THETA) .* sinc_a .* sinc_b).^2;

% Перевод в дБм^2
sigma_plate_dB = 10 * log10(sigma_plate + 1e-30);  % +eps для избежания log(0)

% ---- 3D-график ----
figure('Name', 'Задание 3 — ЭПР прямоугольной пластины', 'NumberTitle', 'off');

subplot(2,2,1);
surf(THETA*180/pi, PHI*180/pi, sigma_plate_dB, 'EdgeColor', 'none');
shading interp; colormap(jet); colorbar;
xlabel('\theta, град'); ylabel('\phi, град');
zlabel('\sigma, дБм^2');
title('3D: ЭПР пластины (дБм^2)');
view(45, 30);

% ---- Сечение xz (phi = 0) ----
phi_xz = 0;
arg_a_xz = 2 * k * a_plate * sin(theta_plate) * cos(phi_xz);
arg_b_xz = 2 * k * b_plate * sin(theta_plate) * sin(phi_xz);

sinc_a_xz = ones(size(arg_a_xz));
sinc_b_xz = ones(size(arg_b_xz));
mask_a_xz = abs(arg_a_xz) > 1e-10;
mask_b_xz = abs(arg_b_xz) > 1e-10;
sinc_a_xz(mask_a_xz) = sin(arg_a_xz(mask_a_xz)) ./ arg_a_xz(mask_a_xz);
sinc_b_xz(mask_b_xz) = sin(arg_b_xz(mask_b_xz)) ./ arg_b_xz(mask_b_xz);

sigma_xz = sigma_m_plate * (cos(theta_plate) .* sinc_a_xz .* sinc_b_xz).^2;
sigma_xz_dB = 10*log10(sigma_xz + 1e-30);

subplot(2,2,2);
plot(theta_plate*180/pi, sigma_xz_dB, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('\theta, град'); ylabel('\sigma, дБм^2');
title('Сечение xz: \phi = 0°');
xlim([0 90]); ylim([-60 max(sigma_xz_dB)+5]);

% ---- Сечение yz (phi = pi/2) ----
phi_yz = pi/2;
arg_a_yz = 2 * k * a_plate * sin(theta_plate) * cos(phi_yz);
arg_b_yz = 2 * k * b_plate * sin(theta_plate) * sin(phi_yz);

sinc_a_yz = ones(size(arg_a_yz));
sinc_b_yz = ones(size(arg_b_yz));
mask_a_yz = abs(arg_a_yz) > 1e-10;
mask_b_yz = abs(arg_b_yz) > 1e-10;
sinc_a_yz(mask_a_yz) = sin(arg_a_yz(mask_a_yz)) ./ arg_a_yz(mask_a_yz);
sinc_b_yz(mask_b_yz) = sin(arg_b_yz(mask_b_yz)) ./ arg_b_yz(mask_b_yz);

sigma_yz = sigma_m_plate * (cos(theta_plate) .* sinc_a_yz .* sinc_b_yz).^2;
sigma_yz_dB = 10*log10(sigma_yz + 1e-30);

subplot(2,2,3);
plot(theta_plate*180/pi, sigma_yz_dB, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('\theta, град'); ylabel('\sigma, дБм^2');
title('Сечение yz: \phi = 90°');
xlim([0 90]); ylim([-60 max(sigma_yz_dB)+5]);

% ---- Совмещённое сечение ----
subplot(2,2,4);
plot(theta_plate*180/pi, sigma_xz_dB, 'b-', 'LineWidth', 1.5); hold on;
plot(theta_plate*180/pi, sigma_yz_dB, 'r--', 'LineWidth', 1.5);
grid on;
xlabel('\theta, град'); ylabel('\sigma, дБм^2');
title('Сравнение сечений xz и yz');
legend('\phi = 0°', '\phi = 90°', 'Location', 'best');
xlim([0 90]); ylim([-60 max([sigma_xz_dB, sigma_yz_dB])+5]);


%% =========================================================================
%%  ЗАДАНИЕ 4 — ЭПР КОНИЧЕСКОГО ОТРАЖАТЕЛЯ
%%
%%  Угол раскрыва alpha от 10° до 70°
%%  Размер a домножить на 10
%%  Формула (8): sigma_m = pi * a^2 * tg^2(alpha)
%%
%%  Примечание: ЭПР конуса максимальна при наблюдении вдоль оси конуса
%%  и определяется радиусом основания и углом при вершине.
%% =========================================================================

fprintf('--- Задание 4: ЭПР конуса ---\n');

% Размер основания конуса (домножен на 10)
a_cone = a * 10;   % 0.1 м

fprintf('  Радиус основания конуса a = %.1f мм\n', a_cone*1e3);

% Диапазон углов раскрыва конуса (полный угол при вершине)
alpha_cone = linspace(10, 70, 500);     % градусы
alpha_rad  = alpha_cone * pi / 180;     % радианы

% Формула (8): ЭПР конуса
sigma_cone = pi * a_cone^2 * tan(alpha_rad).^2;   % м^2

% Перевод в дБ
sigma_cone_dB = 10 * log10(sigma_cone);

% Построение графика
figure('Name', 'Задание 4 — ЭПР конуса', 'NumberTitle', 'off');

subplot(2,1,1);
plot(alpha_cone, sigma_cone, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('\alpha, градусы');
ylabel('\sigma, м^2');
title(['ЭПР конуса: \sigma = \pi a^2 tg^2(\alpha), a = ', ...
       num2str(a_cone*1e3, '%.0f'), ' мм']);

subplot(2,1,2);
plot(alpha_cone, sigma_cone_dB, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('\alpha, градусы');
ylabel('\sigma, дБм^2');
title('ЭПР конуса в децибелах');

% Контрольные значения
fprintf('  sigma(10°)  = %.4f м^2 (%.1f дБм^2)\n', ...
    sigma_cone(1), sigma_cone_dB(1));
fprintf('  sigma(45°)  = %.4f м^2 (%.1f дБм^2)\n', ...
    sigma_cone(round((45-10)/60*(length(alpha_cone)-1)+1)), ...
    sigma_cone_dB(round((45-10)/60*(length(alpha_cone)-1)+1)));
fprintf('  sigma(70°)  = %.4f м^2 (%.1f дБм^2)\n\n', ...
    sigma_cone(end), sigma_cone_dB(end));


%% =========================================================================
%%  ЗАДАНИЕ 5 — ЭПР ДВУГРАННОГО УГОЛКОВОГО ОТРАЖАТЕЛЯ
%%
%%  Двугранный уголковый отражатель с углом 90° между гранями.
%%  Размеры граней: a x b.
%%
%%  (а) В плоскости, перпендикулярной ребру — формула (9):
%%      sigma(phi) = 2*sigma_m * cos^2(2*phi) * sinc^2(2*ka*sin(phi))
%%      sigma_m = 8*pi*a^2*b^2 / lambda^2
%%
%%  (б) В ортогональной плоскости — формула (10):
%%      sigma(theta) = (8*pi*a^2*b^2 / lambda^2) * sin(theta) *
%%                     [sin(kb*sin(theta)) / (kb*sin(theta))]^2
%% =========================================================================

fprintf('--- Задание 5: ЭПР двугранного уголкового отражателя ---\n');

% Максимальная ЭПР двугранного отражателя
sigma_m_dihedral = 8 * pi * a^2 * b^2 / lambda^2;
fprintf('  sigma_m = %.3e м^2 (%.1f дБм^2)\n', sigma_m_dihedral, ...
    10*log10(sigma_m_dihedral));

% ---- (а) Плоскость, перпендикулярная ребру (phi) ----
phi_dihedral = linspace(-pi/4, pi/4, 1000);  % от -45° до +45°

% Формула (9): ЭПР двугранного отражателя
% sigma(phi) = 2 * sigma_m * cos^2(2*phi) * sinc^2(2*ka*sin(phi))
%
% Аргумент sinc-функции
arg_dih = 2 * k * a * sin(phi_dihedral);
sinc_dih = ones(size(arg_dih));
mask_dih = abs(arg_dih) > 1e-10;
sinc_dih(mask_dih) = sin(arg_dih(mask_dih)) ./ arg_dih(mask_dih);

sigma_phi = 2 * sigma_m_dihedral * cos(2*phi_dihedral).^2 .* sinc_dih.^2;
sigma_phi_dB = 10*log10(sigma_phi + 1e-30);

% ---- (б) Ортогональная плоскость (theta) ----
theta_dihedral = linspace(0.001, pi/2, 1000);  % избегаем theta=0

% Формула (10): ЭПР в ортогональной плоскости
arg_dih_b = k * b * sin(theta_dihedral);
sinc_dih_b = ones(size(arg_dih_b));
mask_dih_b = abs(arg_dih_b) > 1e-10;
sinc_dih_b(mask_dih_b) = sin(arg_dih_b(mask_dih_b)) ./ arg_dih_b(mask_dih_b);

sigma_theta = sigma_m_dihedral * sin(theta_dihedral) .* sinc_dih_b.^2;
sigma_theta_dB = 10*log10(sigma_theta + 1e-30);

% Построение графиков
figure('Name', 'Задание 5 — ЭПР двугранного отражателя', 'NumberTitle', 'off');

subplot(2,2,1);
plot(phi_dihedral*180/pi, sigma_phi, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('\phi, градусы');
ylabel('\sigma, м^2');
title('(а) Плоскость \perp ребру (линейный)');

subplot(2,2,2);
plot(phi_dihedral*180/pi, sigma_phi_dB, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('\phi, градусы');
ylabel('\sigma, дБм^2');
title('(а) Плоскость \perp ребру (дБ)');

subplot(2,2,3);
plot(theta_dihedral*180/pi, sigma_theta, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('\theta, градусы');
ylabel('\sigma, м^2');
title('(б) Ортогональная плоскость (линейный)');

subplot(2,2,4);
plot(theta_dihedral*180/pi, sigma_theta_dB, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('\theta, градусы');
ylabel('\sigma, дБм^2');
title('(б) Ортогональная плоскость (дБ)');

fprintf('  sigma_phi(0)  = %.3e м^2 (биссектриса)\n', sigma_phi(length(phi_dihedral)/2+1));
fprintf('  sigma_theta   max = %.3e м^2\n\n', max(sigma_theta));


%% =========================================================================
%%  ЗАДАНИЕ 6 — МАКСИМАЛЬНЫЕ ЭПР ТРЁХГРАННЫХ УГОЛКОВЫХ ОТРАЖАТЕЛЕЙ
%%
%%  Зависимость максимальных ЭПР от длины ребра a при фиксированной lambda.
%%  Диапазон: a от 1*lambda до 20*lambda.
%%  Типы граней:
%%    Формула (11):
%%      sigma_tr = 4*pi * a^4 / (3*lambda^2)    — треугольный
%%      sigma_pr = 12*pi * a^4 / lambda^2        — прямоугольный
%%      sigma_cr = 2*pi * a^4 / lambda^2         — круглый (секторный)
%% =========================================================================

fprintf('--- Задание 6: ЭПР трёхгранных отражателей ---\n');

% Диапазон длин рёбер
a_tri = linspace(1*lambda, 20*lambda, 500);  % от lambda до 20*lambda

% Формула (11): максимальные ЭПР трёхгранных отражателей
sigma_triangular  = (4/3) * pi * a_tri.^4 / lambda^2;   % треугольный
sigma_rectangular = 12 * pi * a_tri.^4 / lambda^2;       % прямоугольный
sigma_circular    = 2 * pi * a_tri.^4 / lambda^2;        % секторный (круглый)

% Перевод в дБ
sigma_triangular_dB  = 10*log10(sigma_triangular);
sigma_rectangular_dB = 10*log10(sigma_rectangular);
sigma_circular_dB    = 10*log10(sigma_circular);

% Построение графиков
figure('Name', 'Задание 6 — ЭПР трёхгранных отражателей', 'NumberTitle', 'off');

subplot(2,1,1);
plot(a_tri/lambda, sigma_triangular, 'b-', 'LineWidth', 1.5); hold on;
plot(a_tri/lambda, sigma_rectangular, 'r-', 'LineWidth', 1.5);
plot(a_tri/lambda, sigma_circular, 'g-', 'LineWidth', 1.5);
grid on;
xlabel('a / \lambda');
ylabel('\sigma, м^2');
title('Максимальные ЭПР трёхгранных отражателей');
legend('Треугольный (4\pi a^4 / 3\lambda^2)', ...
       'Прямоугольный (12\pi a^4 / \lambda^2)', ...
       'Секторный (2\pi a^4 / \lambda^2)', 'Location', 'best');

subplot(2,1,2);
plot(a_tri/lambda, sigma_triangular_dB, 'b-', 'LineWidth', 1.5); hold on;
plot(a_tri/lambda, sigma_rectangular_dB, 'r-', 'LineWidth', 1.5);
plot(a_tri/lambda, sigma_circular_dB, 'g-', 'LineWidth', 1.5);
grid on;
xlabel('a / \lambda');
ylabel('\sigma, дБм^2');
title('Максимальные ЭПР трёхгранных отражателей (дБ)');
legend('Треугольный', 'Прямоугольный', 'Секторный', 'Location', 'best');

% Вывод таблицы для нескольких значений
fprintf('\n  ЭПР при различных размерах ребра:\n');
fprintf('  %-10s %-15s %-15s %-15s\n', 'a/lambda', 'Треуг.(м^2)', 'Прям.(м^2)', 'Сект.(м^2)');
fprintf('  %-10s %-15s %-15s %-15s\n', '--------', '-------------', '-------------', '-------------');
idx = [1, round(length(a_tri)/4), round(length(a_tri)/2), ...
       round(3*length(a_tri)/4), length(a_tri)];
for ii = idx
    fprintf('  %-10.1f %-15.3e %-15.3e %-15.3e\n', ...
        a_tri(ii)/lambda, sigma_triangular(ii), sigma_rectangular(ii), sigma_circular(ii));
end
fprintf('\n');


%% =========================================================================
%%  ЗАДАНИЕ 7 — ЭПР ЛИНЗЫ ЛЮНЕБЕРГА С МЕТАЛЛИЧЕСКИМ КОЛЬЦОМ
%%
%%  Радиус линзы: R = 20 * lambda
%%  Ширина кольца: l от 0.1*R до 1.8*R
%%  Формула (13):
%%    sigma = (4*pi / lambda^2) * (2*R*l - l^2)^2
%%
%%  Примечание: при l = R получаем максимальную ЭПР,
%%  совпадающую с формулой (12) для полусферической линзы:
%%  sigma_max = 4*pi*R^4 / lambda^2
%% =========================================================================

fprintf('--- Задание 7: ЭПР линзы Люнеберга с кольцом ---\n');

% Параметры линзы Люнеберга
R_lens = 20 * lambda;    % Радиус линзы, м
fprintf('  Радиус линзы R = %.3f м (%.1f*lambda)\n', R_lens, R_lens/lambda);

% Диапазон ширины металлизированного кольца
l_ring = linspace(0.1*R_lens, 1.8*R_lens, 500);   % от 0.1R до 1.8R

% Формула (13): ЭПР линзы Люнеберга с кольцом
% Эффективная отражающая площадь: A_eff = l*(2R - l) = 2Rl - l^2
A_eff = 2 * R_lens * l_ring - l_ring.^2;
sigma_luneburg = (4 * pi / lambda^2) * A_eff.^2;   % м^2

% Проверка: при l = R должно совпадать с формулой (12)
sigma_max_luneburg = 4 * pi * R_lens^4 / lambda^2;
fprintf('  sigma_max (формула 12, l=R): %.3e м^2\n', sigma_max_luneburg);
fprintf('  sigma при l=R (расчёт):      %.3e м^2\n', ...
    (4*pi/lambda^2) * (2*R_lens*R_lens - R_lens^2)^2);

% Перевод в дБ
sigma_luneburg_dB = 10*log10(sigma_luneburg);

% Построение графика
figure('Name', 'Задание 7 — ЭПР линзы Люнеберга', 'NumberTitle', 'off');

subplot(2,1,1);
plot(l_ring/R_lens, sigma_luneburg, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('l / R');
ylabel('\sigma, м^2');
title(['ЭПР линзы Люнеберга: R = ', num2str(R_lens, '%.2f'), ...
       ' м (20\lambda), \lambda = ', num2str(lambda*1e3, '%.0f'), ' мм']);
% Отметим максимум
[~, idx_max] = max(sigma_luneburg);
hold on;
plot(l_ring(idx_max)/R_lens, sigma_luneburg(idx_max), 'ro', ...
    'MarkerSize', 8, 'LineWidth', 2);
text(l_ring(idx_max)/R_lens + 0.05, sigma_luneburg(idx_max), ...
    sprintf('Максимум: l/R = %.2f', l_ring(idx_max)/R_lens));

subplot(2,1,2);
plot(l_ring/R_lens, sigma_luneburg_dB, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('l / R');
ylabel('\sigma, дБм^2');
title('ЭПР линзы Люнеберга (дБ)');

fprintf('  sigma при l = 0.1R: %.3e м^2 (%.1f дБм^2)\n', ...
    sigma_luneburg(1), sigma_luneburg_dB(1));
fprintf('  sigma при l = R:    %.3e м^2 (%.1f дБм^2)\n', ...
    sigma_luneburg(idx_max), sigma_luneburg_dB(idx_max));
fprintf('  sigma при l = 1.8R: %.3e м^2 (%.1f дБм^2)\n\n', ...
    sigma_luneburg(end), sigma_luneburg_dB(end));


%% =========================================================================
%%  ЗАДАНИЕ 8 — ЭПР РЕШЁТКИ ВАН-АТТА
%%
%%  Число диполей: N = 64 * номер_варианта
%%  Угол падения: theta от 0° до 80°
%%  Формула (14):
%%    sigma(theta) = (4*pi*S^2 / lambda^2) * sin^4((pi/2) * cos(theta))
%%  где S = N*lambda^2 / 4 — площадь раскрыва решётки
%%
%%  При theta = 0: sigma_max = pi*lambda^2*N^2/4
%% =========================================================================

fprintf('--- Задание 8: ЭПР решётки Ван-Атта ---\n');

% Число диполей
N_dipoles = 64 * variant;
fprintf('  Число диполей N = %d\n', N_dipoles);

% Площадь раскрыва решётки
S_va = N_dipoles * lambda^2 / 4;   % м^2
fprintf('  Площадь раскрыва S = N*lambda^2/4 = %.4f м^2\n', S_va);

% Максимальная ЭПР (при theta = 0)
sigma_max_va = pi * lambda^2 * N_dipoles^2 / 4;
fprintf('  sigma_max (theta=0) = pi*lambda^2*N^2/4 = %.3e м^2 (%.1f дБм^2)\n', ...
    sigma_max_va, 10*log10(sigma_max_va));

% Углы падения
theta_va = linspace(0, 80*pi/180, 500);  % от 0° до 80°

% Формула (14): ЭПР решётки Ван-Атта
sigma_va = (4 * pi * S_va^2 / lambda^2) * sin((pi/2) * cos(theta_va)).^4;

% Перевод в дБ
sigma_va_dB = 10*log10(sigma_va);

% Построение графика
figure('Name', 'Задание 8 — ЭПР решётки Ван-Атта', 'NumberTitle', 'off');

subplot(2,1,1);
plot(theta_va*180/pi, sigma_va, 'b-', 'LineWidth', 1.5);
grid on;
xlabel('\theta, градусы');
ylabel('\sigma, м^2');
title(['ЭПР решётки Ван-Атта: N = ', num2str(N_dipoles), ...
       ', f = ', num2str(f/1e9, '%.1f'), ' ГГц']);
xlim([0 80]);

subplot(2,1,2);
plot(theta_va*180/pi, sigma_va_dB, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('\theta, градусы');
ylabel('\sigma, дБм^2');
title('ЭПР решётки Ван-Атта (дБ)');
xlim([0 80]); ylim([min(sigma_va_dB)-5 max(sigma_va_dB)+5]);

% Контрольные значения
fprintf('  sigma(0)  = %.3e м^2 (%.1f дБм^2)\n', sigma_va(1), sigma_va_dB(1));
idx_30 = round(30/80*(length(theta_va)-1))+1;
idx_60 = round(60/80*(length(theta_va)-1))+1;
fprintf('  sigma(30) = %.3e м^2 (%.1f дБм^2)\n', sigma_va(idx_30), sigma_va_dB(idx_30));
fprintf('  sigma(60) = %.3e м^2 (%.1f дБм^2)\n', sigma_va(idx_60), sigma_va_dB(idx_60));
fprintf('  sigma(80) = %.3e м^2 (%.1f дБм^2)\n\n', sigma_va(end), sigma_va_dB(end));


%% =========================================================================
%%  СВОДНАЯ ИНФОРМАЦИЯ
%% =========================================================================

fprintf('\n==========================================================\n');
fprintf('  Все задания выполнены. Общее количество фигур: 8\n');
fprintf('==========================================================\n');
fprintf('  Рекомендации:\n');
fprintf('  - Используйте меню Figure для навигации по графикам\n');
fprintf('  - Для смены варианта измените параметры в БЛОКЕ 0\n');
fprintf('  - Все формулы соответствуют методическим указаниям\n');
fprintf('==========================================================\n');
