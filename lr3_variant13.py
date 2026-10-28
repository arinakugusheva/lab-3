"""
Лабораторная работа № 3. Вариант 13.
Дисциплина: Теория алгоритмов.
Выполнила: Кугушева Арина Вячеславовна, группа БИ 2-1.

Алгоритмы «разделяй и властвуй»: быстрая сортировка и сортировка слиянием.

Исходные данные варианта 13:
    опорный элемент - средний (С);
    схема разбиения - Ломуто (Л);
    сортировка слиянием - восходящая, итеративная (В);
    размеры для эксперимента 2: 10 000; 40 000; 70 000; 100 000;
    дополнительное задание Г1 - гибридная быстрая сортировка:
    для подмассивов длиной <= M применяется сортировка вставками,
    оптимальное M выбирается из {5, 10, 16, 32, 64} экспериментально.

Эксперименты:
    1 - четыре алгоритма (пузырьком, вставками, быстрая, слиянием) на
        размерах из варианта 13 ЛР № 2 (n = 1000..5000);
    2 - быстрая, слиянием, гибридная быстрая и sorted() на больших n;
    3 - быстрая и слиянием на данных R, S, V, N, D при n = 50 000.

Файл самодостаточен (функции ЛР № 2 включены сюда же).
Запуск:  python lr3_variant13.py          (график сохраняется в lr3_plot.png)
         python lr3_variant13.py --show   (дополнительно показать график)
Полный прогон занимает около минуты (быстрая сортировка по схеме Ломуто
на данных типа D работает квадратично - это ожидаемый результат).
Требуется: matplotlib.
"""
import json
import math
import platform
import random
import statistics
import sys
import time

import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")          # без окна: график только сохраняется в файл
import matplotlib.pyplot as plt

# ----------------------------- параметры варианта -----------------------------
SIZES_1 = [1000, 2000, 3000, 4000, 5000]          # размеры ЛР № 2 (вариант 13)
SIZES_2 = [10_000, 40_000, 70_000, 100_000]       # эксперимент 2 (вариант 13)
SIZES_2B = [12_500, 25_000, 50_000, 100_000]      # удвоение n для T(2n)/T(n)
N_3 = 50_000                                      # эксперимент 3
M_VALUES = [5, 10, 16, 32, 64]                    # Г1: кандидаты на порог M
N_G1 = 100_000                                    # размер массива для подбора M
REPEATS = 5
REPEATS_G1 = 7                                    # для подбора M (разница малая)
SEED = 42


# ------------------------------ генерация данных ------------------------------
def generate_data(n, kind="random", lo=0, hi=100_000, seed=SEED):
    """Массив длины n: random (R), sorted (S), reversed (V),
    nearly_sorted (N, n/20 случайных перестановок), dup (D, значения 0..10)."""
    rng = random.Random(seed + n)
    if kind == "dup":
        return [rng.randint(0, 10) for _ in range(n)]
    data = [rng.randint(lo, hi) for _ in range(n)]
    if kind == "sorted":
        data.sort()
    elif kind == "reversed":
        data.sort(reverse=True)
    elif kind == "nearly_sorted":
        data.sort()
        for _ in range(max(1, n // 20)):
            i, j = rng.randrange(n), rng.randrange(n)
            data[i], data[j] = data[j], data[i]
    return data


# ---------------------- алгоритмы ЛР № 2 (квадратичные) -----------------------
def bubble_sort(arr):
    """Сортировка пузырьком с флагом досрочного выхода."""
    a = arr.copy()
    n = len(a)
    for i in range(n - 1):
        swapped = False
        for j in range(n - 1 - i):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:
            break
    return a


def insertion_sort(arr):
    """Сортировка вставками."""
    a = arr.copy()
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a


# ----------------------------- быстрая сортировка -----------------------------
def quick_sort(arr):
    """Быстрая сортировка: опорный - средний, разбиение Ломуто."""
    a = arr.copy()
    _quick_sort(a, 0, len(a) - 1)
    return a


def _quick_sort(a, lo, hi):
    while lo < hi:
        p = _partition_lomuto(a, lo, hi)
        # рекурсия в меньшую часть - глубина стека O(log n)
        if p - lo < hi - p:
            _quick_sort(a, lo, p - 1)
            lo = p + 1
        else:
            _quick_sort(a, p + 1, hi)
            hi = p - 1


def _partition_lomuto(a, lo, hi):
    """Разбиение Ломуто; опорный (средний) элемент переносится в конец."""
    mid = (lo + hi) // 2
    a[mid], a[hi] = a[hi], a[mid]
    pivot = a[hi]
    i = lo - 1
    for j in range(lo, hi):
        if a[j] <= pivot:
            i += 1
            a[i], a[j] = a[j], a[i]
    a[i + 1], a[hi] = a[hi], a[i + 1]
    return i + 1                    # итоговая позиция опорного элемента


# ---------------------- Г1: гибридная быстрая сортировка ----------------------
def hybrid_quick_sort(arr, m=16):
    """Быстрая сортировка; подмассивы длиной <= m сортируются вставками."""
    a = arr.copy()
    _hybrid(a, 0, len(a) - 1, m)
    return a


def _hybrid(a, lo, hi, m):
    while hi - lo + 1 > m:
        p = _partition_lomuto(a, lo, hi)
        if p - lo < hi - p:
            _hybrid(a, lo, p - 1, m)
            lo = p + 1
        else:
            _hybrid(a, p + 1, hi, m)
            hi = p - 1
    _insertion_range(a, lo, hi)     # короткий подмассив - вставками


def _insertion_range(a, lo, hi):
    """Сортировка вставками подмассива a[lo..hi] на месте."""
    for i in range(lo + 1, hi + 1):
        key = a[i]
        j = i - 1
        while j >= lo and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key


# --------------------- сортировка слиянием (восходящая) -----------------------
def merge_sort(arr):
    """Восходящая (итеративная) сортировка слиянием: блоки 1, 2, 4, 8, ..."""
    n = len(arr)
    src = arr.copy()
    dst = [0] * n
    width = 1
    while width < n:
        for lo in range(0, n, 2 * width):
            mid = min(lo + width, n)
            hi = min(lo + 2 * width, n)
            i, j, k = lo, mid, lo
            while i < mid and j < hi:
                if src[i] <= src[j]:         # «<=» обеспечивает устойчивость
                    dst[k] = src[i]
                    i += 1
                else:
                    dst[k] = src[j]
                    j += 1
                k += 1
            if i < mid:                      # остаток левого блока
                dst[k:hi] = src[i:mid]
            elif j < hi:                     # остаток правого блока
                dst[k:hi] = src[j:hi]
        src, dst = dst, src                  # результат слияния становится входом
        width *= 2
    return src


# ------------------------------- замер времени --------------------------------
def measure(sort_func, data, repeats=REPEATS):
    """Медианное время работы sort_func на данных data, с (perf_counter)."""
    expected = sorted(data)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = sort_func(data)
        times.append(time.perf_counter() - start)
        assert result == expected, f"{sort_func.__name__}: ошибка сортировки"
    return statistics.median(times)


def run_table(algorithms, sizes, kind="random", repeats=REPEATS, scale=1.0):
    """Время каждого алгоритма для каждого n; печатает таблицу.
    Результаты возвращаются в секундах; scale=1e3 - печать в миллисекундах."""
    res = {name: [] for name in algorithms}
    print(f"{'n':>8}" + "".join(f"{name:>14}" for name in algorithms))
    for n in sizes:
        data = generate_data(n, kind)
        row = f"{n:>8}"
        for name, func in algorithms.items():
            t = measure(func, data, repeats)
            res[name].append(t)
            row += f"{t * scale:>14.4f}"
        print(row, flush=True)
    return res


def system_info():
    """Характеристики вычислительной системы (для отчёта)."""
    cpu = platform.processor() or platform.machine()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return (f"ОС: {platform.system()} {platform.release()}; процессор: {cpu}; "
            f"Python {platform.python_version()}")


def theory_ratio(n1, n2):
    """Теоретическое T(n2)/T(n1) для Θ(n log n)."""
    return (n2 * math.log2(n2)) / (n1 * math.log2(n1))


# ------------------------------------ main ------------------------------------
def main():
    out = {}
    print("Лабораторная работа № 3, вариант 13")
    print(system_info())
    print("Опорный: средний; разбиение: Ломуто; слияние: восходящее; "
          f"k = {REPEATS}\n")

    # --- Г1: подбор порога M гибридной сортировки (случайные данные)
    print(f"Г1. Подбор M, случайные данные, n = {N_G1}, время, с")
    data = generate_data(N_G1, "random")
    g1 = {"M=1 (без вставок)": measure(quick_sort, data, REPEATS_G1)}
    for m in M_VALUES:
        g1[f"M={m}"] = measure(lambda a, m=m: hybrid_quick_sort(a, m), data,
                               REPEATS_G1)
    for name, t in g1.items():
        print(f"{name:>20}{t:>10.4f}")
    best_m = min(M_VALUES, key=lambda m: g1[f"M={m}"])
    print(f"Оптимальное M = {best_m}\n")
    out["g1"] = g1
    out["best_m"] = best_m

    def hybrid(a):
        return hybrid_quick_sort(a, best_m)
    hybrid.__name__ = "hybrid_quick_sort"

    # --- Эксперимент 1: четыре алгоритма
    algs1 = {"Пузырьком": bubble_sort, "Вставками": insertion_sort,
             "Быстрая": quick_sort, "Слиянием": merge_sort}
    print("Эксперимент 1а. Четыре алгоритма, СЛУЧАЙНЫЕ данные, с")
    res1r = run_table(algs1, SIZES_1, "random")
    print("\nЭксперимент 1б. Четыре алгоритма, УПОРЯДОЧЕННЫЕ данные (тип S "
          "варианта 13 ЛР № 2), мс")
    res1s = run_table(algs1, SIZES_1, "sorted", scale=1e3)
    out["exp1_random"] = res1r
    out["exp1_sorted"] = res1s

    # --- Эксперимент 2: O(n log n) на больших размерах
    algs2 = {"Быстрая": quick_sort, "Слиянием": merge_sort,
             f"Гибрид M={best_m}": hybrid, "sorted()": sorted}
    print("\nЭксперимент 2. Большие массивы, СЛУЧАЙНЫЕ данные, с")
    res2 = run_table(algs2, SIZES_2, "random")
    out["exp2"] = res2
    print("\nОтношение времён (слиянием / быстрая):")
    for n, tq, tm in zip(SIZES_2, res2["Быстрая"], res2["Слиянием"]):
        print(f"{n:>8}{tm / tq:>8.2f}")
    print("\nОтношения T(n2)/T(n1) для соседних размеров эксперимента 2")
    print(f"{'Переход':>20}{'Быстрая':>10}{'Слиянием':>10}"
          f"{'Теор.nlogn':>12}{'Теор.n^2':>10}")
    ratios2 = []
    for i in range(1, len(SIZES_2)):
        n1, n2 = SIZES_2[i - 1], SIZES_2[i]
        rq = res2["Быстрая"][i] / res2["Быстрая"][i - 1]
        rm = res2["Слиянием"][i] / res2["Слиянием"][i - 1]
        th, th2 = theory_ratio(n1, n2), (n2 / n1) ** 2
        ratios2.append([n1, n2, rq, rm, th, th2])
        print(f"{str(n1) + '->' + str(n2):>20}{rq:>10.2f}{rm:>10.2f}"
              f"{th:>12.2f}{th2:>10.2f}")
    out["ratios2"] = ratios2

    print("\nЭксперимент 2б. Удвоение n, СЛУЧАЙНЫЕ данные, с")
    res2b = run_table({"Быстрая": quick_sort, "Слиянием": merge_sort},
                      SIZES_2B, "random")
    print("\nT(2n)/T(n) при удвоении")
    print(f"{'Переход':>20}{'Быстрая':>10}{'Слиянием':>10}{'Теор.nlogn':>12}")
    ratios2b = []
    for i in range(1, len(SIZES_2B)):
        n1, n2 = SIZES_2B[i - 1], SIZES_2B[i]
        rq = res2b["Быстрая"][i] / res2b["Быстрая"][i - 1]
        rm = res2b["Слиянием"][i] / res2b["Слиянием"][i - 1]
        ratios2b.append([n1, n2, rq, rm, theory_ratio(n1, n2)])
        print(f"{str(n1) + '->' + str(n2):>20}{rq:>10.2f}{rm:>10.2f}"
              f"{theory_ratio(n1, n2):>12.2f}")
    out["exp2b"] = res2b
    out["ratios2b"] = ratios2b

    # --- Эксперимент 3: структура данных
    kinds = [("random", "R"), ("sorted", "S"), ("reversed", "V"),
             ("nearly_sorted", "N"), ("dup", "D")]
    print(f"\nЭксперимент 3. Влияние структуры данных, n = {N_3}, с")
    print(f"{'Тип':>6}{'Быстрая':>12}{'Слиянием':>12}{'Слиянием/Быстрая':>20}")
    exp3 = {}
    for kind, code in kinds:
        data = generate_data(N_3, kind)
        tq = measure(quick_sort, data)
        tm = measure(merge_sort, data)
        exp3[code] = [tq, tm]
        print(f"{code:>6}{tq:>12.4f}{tm:>12.4f}{tm / tq:>20.2f}", flush=True)
    out["exp3"] = exp3

    # --- Сводная таблица (n = 5000, случайные данные)
    i5 = SIZES_1.index(5000)
    t_bubble = res1r["Пузырьком"][i5]
    print("\nСводная таблица, n = 5000, случайные данные")
    print(f"{'Алгоритм':>12}{'Время, с':>12}{'Ускорение':>12}")
    for name in algs1:
        t = res1r[name][i5]
        print(f"{name:>12}{t:>12.4f}{t_bubble / t:>12.1f}")

    # --- Графики
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for name, t in res1r.items():
        axes[0, 0].plot(SIZES_1, t, marker="o", label=name)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Эксперимент 1 (случайные данные, лог. шкала)")
    for name, t in res2.items():
        axes[0, 1].plot(SIZES_2, t, marker="o", label=name)
    axes[0, 1].set_title("Эксперимент 2 (большие массивы)")
    for ax in axes[0]:
        ax.set_xlabel("Размер массива n")
        ax.set_ylabel("Время, с")
        ax.grid(True)
        ax.legend()
    codes = [c for _, c in kinds]
    xs = range(len(codes))
    axes[1, 0].bar([x - 0.2 for x in xs], [exp3[c][0] for c in codes], 0.4,
                   label="Быстрая")
    axes[1, 0].bar([x + 0.2 for x in xs], [exp3[c][1] for c in codes], 0.4,
                   label="Слиянием")
    axes[1, 0].set_xticks(list(xs))
    axes[1, 0].set_xticklabels(codes)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title(f"Эксперимент 3 (n = {N_3}, лог. шкала)")
    axes[1, 0].set_xlabel("Тип данных")
    axes[1, 0].set_ylabel("Время, с")
    axes[1, 0].legend()
    axes[1, 0].grid(True, axis="y")
    axes[1, 1].plot([1] + M_VALUES, list(g1.values()), marker="o")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_title(f"Г1: подбор M (n = {N_G1})")
    axes[1, 1].set_xlabel("Порог M (M = 1 — без вставок)")
    axes[1, 1].set_ylabel("Время, с")
    axes[1, 1].grid(True)
    fig.tight_layout()
    fig.savefig("lr3_plot.png", dpi=150)
    print("\nГрафик сохранён в файл lr3_plot.png")

    if "--json" in sys.argv:                     # служебный вывод для отчёта
        with open(sys.argv[sys.argv.index("--json") + 1], "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    if "--show" in sys.argv:
        plt.show()


if __name__ == "__main__":
    main()
