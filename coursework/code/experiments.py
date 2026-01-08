"""
Градиентный спуск при больших шагах: хаос и фрактальная область сходимости
Курсовая работа, 1 курс

Реализация экспериментов из статьи Liang & Montufar (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['figure.dpi'] = 150


# =============================================================================
# ЧАСТЬ 1: СКАЛЯРНАЯ ФАКТОРИЗАЦИЯ
# =============================================================================

def grad_scalar(u, v, y, lam=0.0):
    """
    Градиент для скалярной факторизации
    L(u,v) = (uv - y)^2/2 + lam/2*(u^2 + v^2)
    """
    r = u * v - y
    du = r * v + lam * u
    dv = r * u + lam * v
    return du, dv


def gd_step_scalar(u, v, eta, y, lam=0.0):
    """Один шаг градиентного спуска"""
    du, dv = grad_scalar(u, v, y, lam)
    return u - eta * du, v - eta * dv


def run_gd_scalar(u0, v0, eta, y, lam=0.0, max_iter=1000):
    """
    Запускаем GD из точки (u0, v0)
    Возвращаем: номер аттрактора, конечные координаты
    """
    u, v = u0, v0
    
    for _ in range(max_iter):
        if u*u + v*v > 1e6:
            return 0, u, v  # расходимость
        
        u_new, v_new = gd_step_scalar(u, v, eta, y, lam)
        
        if (u_new - u)**2 + (v_new - v)**2 < 1e-16:
            break
            
        u, v = u_new, v_new
    
    # классификация результата
    if u*u + v*v > 1e6:
        return 0, u, v
    if u*u + v*v < 1e-6:
        return 3, u, v  # седло
    if abs(u*v - y) < 0.1:
        return 1 if u*v > 0 else 2, u, v
    return 1, u, v


def critical_eta(u, v, y):
    """
    Критический шаг из статьи Liang & Montufar
    eta* = min{1/|y|, 8/(||theta||^2 + sqrt(||theta||^4 - 16y(uv-y)))}
    """
    eta1 = 1.0 / abs(y) if y != 0 else 1e10
    
    norm_sq = u*u + v*v
    disc = norm_sq**2 - 16 * y * (u*v - y)
    
    if disc < 0:
        return eta1
    
    eta2 = 8.0 / (norm_sq + np.sqrt(disc))
    return min(eta1, eta2)


# =============================================================================
# ЧАСТЬ 2: МАТРИЧНАЯ ФАКТОРИЗАЦИЯ
# =============================================================================

def grad_matrix(U, V, Y, lam=0.0):
    """
    Градиент для матричной факторизации
    L(U,V) = ||U^T V - Y||_F^2 / 2 + lam/2*(||U||_F^2 + ||V||_F^2)
    """
    R = U.T @ V - Y
    dU = V @ R.T + lam * U
    dV = U @ R + lam * V
    return dU, dV


def run_gd_matrix(U0, V0, eta, Y, lam=0.0, max_iter=500):
    """
    Градиентный спуск для матричной факторизации
    Возвращаем историю loss и финальные матрицы
    """
    U, V = U0.copy(), V0.copy()
    losses = []
    
    for _ in range(max_iter):
        R = U.T @ V - Y
        loss = 0.5 * np.sum(R**2)
        loss += 0.5 * lam * (np.sum(U**2) + np.sum(V**2))
        losses.append(loss)
        
        if loss > 1e8:
            return losses, U, V, False  # расходимость
        
        dU, dV = grad_matrix(U, V, Y, lam)
        U = U - eta * dU
        V = V - eta * dV
        
        if len(losses) > 1 and abs(losses[-1] - losses[-2]) < 1e-12:
            break
    
    return losses, U, V, True


def matrix_basin_slice(Y, eta, lam=0.0, resolution=200, rng=(-2, 2)):
    """
    Срез бассейнов для матричной факторизации
    Меняем только U[0,0] и V[0,0], остальное фиксируем
    """
    d = Y.shape[0]
    
    coords = np.linspace(rng[0], rng[1], resolution)
    basin = np.zeros((resolution, resolution))
    
    for i, alpha in enumerate(coords):
        for j, beta in enumerate(coords):
            U0 = np.eye(d) * 0.1
            V0 = np.eye(d) * 0.1
            U0[0, 0] += alpha
            V0[0, 0] += beta
            
            losses, U, V, ok = run_gd_matrix(U0, V0, eta, Y, lam, max_iter=300)
            
            if not ok:
                basin[j, i] = 0  # расходимость
            else:
                # по норме решения определяем тип
                norm = np.sqrt(np.sum(U**2) + np.sum(V**2))
                basin[j, i] = min(norm, 5)
    
    return basin


# =============================================================================
# ЧАСТЬ 3: ВИЗУАЛИЗАЦИЯ БАССЕЙНОВ
# =============================================================================

def build_basin_map(y, eta, lam=0.0, resolution=400, rng=(-3, 3)):
    """Строим карту бассейнов для скалярного случая"""
    coords = np.linspace(rng[0], rng[1], resolution)
    basin = np.zeros((resolution, resolution), dtype=int)
    
    for i, u0 in enumerate(coords):
        for j, v0 in enumerate(coords):
            basin[j, i], _, _ = run_gd_scalar(u0, v0, eta, y, lam)
    
    return basin


def plot_basins(basin, rng, title, filename):
    """Рисуем карту бассейнов"""
    colors = ['#2c3e50', '#e74c3c', '#3498db', '#f39c12']
    cmap = ListedColormap(colors)
    
    fig, ax = plt.subplots()
    ax.imshow(basin, extent=[rng[0], rng[1], rng[0], rng[1]], 
              origin='lower', cmap=cmap, vmin=0, vmax=3)
    
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    labels = ['Расходимость', 'Минимум (+)', 'Минимум (-)', 'Седло']
    handles = [plt.Rectangle((0,0), 1, 1, facecolor=c) for c in colors]
    ax.legend(handles, labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f'Сохранено: {filename}')


def plot_matrix_basins(basin, rng, title, filename):
    """Рисуем срез бассейнов для матричного случая"""
    fig, ax = plt.subplots()
    
    im = ax.imshow(basin, extent=[rng[0], rng[1], rng[0], rng[1]],
                   origin='lower', cmap='viridis')
    
    ax.set_xlabel('U[0,0]')
    ax.set_ylabel('V[0,0]')
    ax.set_title(title)
    ax.set_aspect('equal')
    
    plt.colorbar(im, ax=ax, label='Норма решения')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f'Сохранено: {filename}')


# =============================================================================
# ЧАСТЬ 4: BOX-COUNTING РАЗМЕРНОСТЬ
# =============================================================================

def find_boundary(basin):
    """Находим границу бассейнов"""
    h, w = basin.shape
    boundary = np.zeros((h, w), dtype=bool)
    
    for i in range(1, h-1):
        for j in range(1, w-1):
            c = basin[i, j]
            if basin[i-1,j]!=c or basin[i+1,j]!=c or basin[i,j-1]!=c or basin[i,j+1]!=c:
                boundary[i, j] = True
    
    return boundary


def box_counting(boundary):
    """Оценка фрактальной размерности"""
    max_size = min(boundary.shape) // 4
    
    sizes = []
    counts = []
    
    box_size = 2
    while box_size <= max_size:
        count = 0
        for i in range(0, boundary.shape[0], box_size):
            for j in range(0, boundary.shape[1], box_size):
                if boundary[i:i+box_size, j:j+box_size].any():
                    count += 1
        
        sizes.append(box_size)
        counts.append(count)
        box_size = int(box_size * 1.5)
    
    sizes = np.array(sizes)
    counts = np.array(counts)
    
    # наклон в log-log координатах
    log_x = np.log(1.0 / sizes)
    log_y = np.log(counts)
    
    mx, my = np.mean(log_x), np.mean(log_y)
    D = np.sum((log_x - mx) * (log_y - my)) / np.sum((log_x - mx)**2)
    
    return D, sizes, counts


def plot_box_counting(sizes, counts, D, filename):
    """График для box-counting"""
    fig, ax = plt.subplots()
    
    log_x = np.log(1.0 / sizes)
    log_y = np.log(counts)
    
    ax.scatter(log_x, log_y, s=50, c='blue', label='Данные')
    
    mx, my = np.mean(log_x), np.mean(log_y)
    x_line = np.linspace(log_x.min(), log_x.max(), 100)
    y_line = D * (x_line - mx) + my
    ax.plot(x_line, y_line, 'r-', linewidth=2, label=f'$D_B$ = {D:.3f}')
    
    ax.set_xlabel('log(1/ε)')
    ax.set_ylabel('log N(ε)')
    ax.set_title('Оценка box-counting размерности')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f'Сохранено: {filename}')


# =============================================================================
# ЧАСТЬ 5: АНАЛИЗ ЧУВСТВИТЕЛЬНОСТИ И СРАВНЕНИЕ С ТЕОРИЕЙ
# =============================================================================

def sensitivity_test(u0, v0, eta, y, lam=0.0, eps=1e-5, n_samples=200):
    """Тест чувствительности к начальным условиям"""
    results = {0: 0, 1: 0, 2: 0, 3: 0}
    norms = []
    
    for _ in range(n_samples):
        du = np.random.uniform(-eps, eps)
        dv = np.random.uniform(-eps, eps)
        
        attr, u_fin, v_fin = run_gd_scalar(u0 + du, v0 + dv, eta, y, lam)
        results[attr] += 1
        norms.append(np.sqrt(u_fin**2 + v_fin**2))
    
    return results, norms


def compare_with_theory(y, lam=0.0, n_points=20):
    """
    Сравниваем экспериментальную границу сходимости с теоретическим eta*
    """
    # точки на окружности
    angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    radii = [0.5, 1.0, 1.5, 2.0]
    
    results = []
    
    for r in radii:
        for theta in angles:
            u0 = r * np.cos(theta)
            v0 = r * np.sin(theta)
            
            # теоретический критический шаг
            eta_theory = critical_eta(u0, v0, y)
            
            # экспериментально находим границу
            eta_exp = 0.1
            while eta_exp < 2.0:
                _, u_fin, v_fin = run_gd_scalar(u0, v0, eta_exp, y, lam, max_iter=500)
                if u_fin**2 + v_fin**2 > 1e5:  # расходимость
                    break
                eta_exp += 0.05
            
            results.append({
                'u0': u0, 'v0': v0, 'r': r,
                'eta_theory': eta_theory,
                'eta_exp': eta_exp
            })
    
    return results


def plot_sensitivity(results, norms, title, filename):
    """Гистограммы чувствительности"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    labels = ['Расх.', 'Мин+', 'Мин-', 'Седло']
    values = [results[i] for i in range(4)]
    colors = ['#2c3e50', '#e74c3c', '#3498db', '#f39c12']
    
    axes[0].bar(labels, values, color=colors)
    axes[0].set_ylabel('Количество')
    axes[0].set_title('Распределение по аттракторам')
    
    axes[1].hist(norms, bins=20, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Норма решения')
    axes[1].set_ylabel('Частота')
    axes[1].set_title('Распределение норм')
    
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f'Сохранено: {filename}')


def plot_theory_comparison(results, filename):
    """График сравнения теории и эксперимента"""
    eta_th = [r['eta_theory'] for r in results]
    eta_ex = [r['eta_exp'] for r in results]
    
    fig, ax = plt.subplots()
    
    ax.scatter(eta_th, eta_ex, alpha=0.6)
    
    # линия y = x
    max_val = max(max(eta_th), max(eta_ex))
    ax.plot([0, max_val], [0, max_val], 'r--', label='Теория = Эксперимент')
    
    ax.set_xlabel('Теоретический η*')
    ax.set_ylabel('Экспериментальный η*')
    ax.set_title('Сравнение теоретического и экспериментального критического шага')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f'Сохранено: {filename}')


# =============================================================================
# ЧАСТЬ 6: ТРАЕКТОРИИ И КРИВЫЕ ПОТЕРЬ
# =============================================================================

def plot_trajectories(eta, y, lam=0.0, n_traj=15, max_iter=100, filename=None):
    """Траектории GD"""
    fig, ax = plt.subplots()
    
    np.random.seed(42)
    
    for _ in range(n_traj):
        u0, v0 = np.random.uniform(-2, 2), np.random.uniform(-2, 2)
        
        traj_u, traj_v = [u0], [v0]
        u, v = u0, v0
        
        for _ in range(max_iter):
            u, v = gd_step_scalar(u, v, eta, y, lam)
            traj_u.append(u)
            traj_v.append(v)
            if u*u + v*v > 50:
                break
        
        ax.plot(traj_u, traj_v, '-', alpha=0.6, linewidth=0.8)
        ax.plot(u0, v0, 'ko', markersize=3)
    
    if y > 0:
        s = np.sqrt(y)
        ax.plot([s, -s], [s, -s], 'r*', markersize=12, label='Минимумы')
    ax.plot(0, 0, 'x', color='orange', markersize=10, label='Седло')
    
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title(f'Траектории GD, η={eta}')
    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        print(f'Сохранено: {filename}')
    plt.close()


def plot_loss_curves(eta_list, y, u0, v0, lam=0.0, max_iter=200, filename=None):
    """Кривые потерь"""
    fig, ax = plt.subplots()
    
    for eta in eta_list:
        u, v = u0, v0
        losses = []
        
        for _ in range(max_iter):
            L = 0.5 * (u*v - y)**2 + 0.5 * lam * (u*u + v*v)
            losses.append(L)
            if L > 1e5:
                break
            u, v = gd_step_scalar(u, v, eta, y, lam)
        
        ax.semilogy(losses, label=f'η={eta}', linewidth=1.5)
    
    ax.set_xlabel('Итерация')
    ax.set_ylabel('Loss')
    ax.set_title('Динамика функции потерь')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        print(f'Сохранено: {filename}')
    plt.close()


def plot_matrix_loss(Y, eta_list, lam=0.0, filename=None):
    """Кривые потерь для матричной факторизации"""
    fig, ax = plt.subplots()
    
    d = Y.shape[0]
    np.random.seed(123)
    U0 = np.random.randn(d, d) * 0.5
    V0 = np.random.randn(d, d) * 0.5
    
    for eta in eta_list:
        losses, _, _, _ = run_gd_matrix(U0.copy(), V0.copy(), eta, Y, lam, max_iter=300)
        ax.semilogy(losses, label=f'η={eta}', linewidth=1.5)
    
    ax.set_xlabel('Итерация')
    ax.set_ylabel('Loss')
    ax.set_title('Матричная факторизация: динамика потерь')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
        print(f'Сохранено: {filename}')
    plt.close()


# =============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# =============================================================================

def main():
    import os
    os.makedirs('../figures', exist_ok=True)
    
    print('=' * 60)
    print('Эксперименты: градиентный спуск при больших шагах')
    print('=' * 60)
    
    y = 1.0
    
    # --- 1. Скалярная факторизация: бассейны ---
    print('\n[1/8] Бассейны сходимости (скалярный случай, λ=0)...')
    for eta in [0.3, 0.5, 0.8, 0.95]:
        basin = build_basin_map(y, eta, lam=0.0, resolution=400)
        plot_basins(basin, (-3, 3), f'Бассейны: η={eta}, λ=0',
                   f'../figures/basin_eta{eta:.2f}_lam0.png')
    
    # --- 2. С регуляризацией ---
    print('\n[2/8] Бассейны с регуляризацией (λ=0.1)...')
    for eta in [0.5, 0.8]:
        basin = build_basin_map(y, eta, lam=0.1, resolution=500)
        plot_basins(basin, (-3, 3), f'Бассейны: η={eta}, λ=0.1',
                   f'../figures/basin_eta{eta:.2f}_lam0.1.png')
    
    # --- 3. Фрактальная размерность ---
    print('\n[3/8] Оценка фрактальной размерности...')
    basin_hr = build_basin_map(y, 0.8, lam=0.1, resolution=800, rng=(-2, 2))
    boundary = find_boundary(basin_hr)
    D, sizes, counts = box_counting(boundary)
    print(f'    Box-counting размерность: D_B = {D:.4f}')
    
    plot_box_counting(sizes, counts, D, '../figures/box_counting.png')
    
    fig, ax = plt.subplots()
    ax.imshow(boundary, extent=[-2, 2, -2, 2], origin='lower', cmap='binary')
    ax.set_xlabel('u')
    ax.set_ylabel('v')
    ax.set_title(f'Граница бассейнов, $D_B$ ≈ {D:.2f}')
    plt.tight_layout()
    plt.savefig('../figures/boundary.png', dpi=300)
    plt.close()
    print('Сохранено: ../figures/boundary.png')
    
    # --- 4. Чувствительность ---
    print('\n[4/8] Анализ чувствительности...')
    results, norms = sensitivity_test(0.5, 0.5, 0.9, y, lam=0.0, eps=1e-4, n_samples=300)
    plot_sensitivity(results, norms, 'Чувствительность: (0.5, 0.5), η=0.9',
                    '../figures/sensitivity.png')
    
    # --- 5. Сравнение с теорией ---
    print('\n[5/8] Сравнение с теоретическим η*...')
    theory_results = compare_with_theory(y, lam=0.0, n_points=15)
    plot_theory_comparison(theory_results, '../figures/theory_comparison.png')
    
    # --- 6. Матричная факторизация ---
    print('\n[6/8] Матричная факторизация...')
    Y_matrix = np.diag([1.0, 0.5])  # диагональная 2x2
    
    # бассейны (срез)
    basin_mat = matrix_basin_slice(Y_matrix, 0.3, lam=0.0, resolution=150, rng=(-2, 2))
    plot_matrix_basins(basin_mat, (-2, 2), 'Матричная факторизация: η=0.3',
                      '../figures/matrix_basin_eta0.3.png')
    
    basin_mat2 = matrix_basin_slice(Y_matrix, 0.5, lam=0.0, resolution=150, rng=(-2, 2))
    plot_matrix_basins(basin_mat2, (-2, 2), 'Матричная факторизация: η=0.5',
                      '../figures/matrix_basin_eta0.5.png')
    
    # кривые потерь
    plot_matrix_loss(Y_matrix, [0.1, 0.2, 0.3, 0.4], lam=0.0,
                    filename='../figures/matrix_loss.png')
    
    # --- 7. Траектории ---
    print('\n[7/8] Траектории...')
    for eta in [0.3, 0.7, 0.95]:
        plot_trajectories(eta, y, filename=f'../figures/trajectories_eta{eta:.2f}.png')
    
    # --- 8. Кривые потерь ---
    print('\n[8/8] Кривые потерь (скалярный случай)...')
    plot_loss_curves([0.2, 0.5, 0.8, 0.95, 1.0], y, 1.5, 0.3,
                    filename='../figures/loss_curves.png')
    
    print('\n' + '=' * 60)
    print('Готово!')
    print('=' * 60)
    
    return D


if __name__ == '__main__':
    dimension = main()
    print(f'\nИтоговая оценка D_B: {dimension:.4f}')
