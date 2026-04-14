import opfunu.cec_based.cec2014 as opf

import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from itertools import product
import sys

from detpy.DETAlgs.data.alg_data import DEData, IDEData, OppBasedData, EIDEData
from detpy.DETAlgs.de import DE
from detpy.DETAlgs.eide import EIDE
from detpy.DETAlgs.ide import IDE
from detpy.DETAlgs.opposition_based import OppBasedDE
from detpy.database.database_connector import SQLiteConnector
from detpy.models.enums.boundary_constrain import BoundaryFixing
from detpy.models.enums.optimization import OptimizationType
from detpy.models.fitness_function import FitnessFunctionOpfunu, FitnessFunctionBase

def cantilever_objective_old(x):
    #Siła skupiona na końcu belki
    P = 50000
    #Moduł Younga
    E = 2e7
    #Długość jednego segmentu
    L = 100
    #Dopuszczalne naprężenie
    sigma_max = 14000
    #Dopuszczalne ugięcie końca
    delta_max = 2.7
    penalty = 0.0
    lambda_pen = 1e6
    # Funkcja celu (objętość)
    f = np.sum(x)
    # Ograniczenia naprężeń
    for i in range(len(x)):
        sigma = P * (len(x)-i) * L / x[i]
        if sigma > sigma_max:
            penalty += (sigma - sigma_max)**2
    # Ograniczenie ugięcia
    delta = (P / E) * np.sum(((np.arange(1,len(x)+1)*L)**3) / x)
    if delta > delta_max:
        penalty += (delta - delta_max)**2

    return f + lambda_pen * penalty

def cantilever_objective(x):

    '''

    optimum
    10 - 334.26
    30 - 976
    50 - 1625
    100 - 3300

    '''
    #Siła skupiona na końcu belki
    P = 50000
    #Moduł Younga
    E = 2e7
    #Długość jednego segmentu
    L = 100 / len(x)
    #Dopuszczalne naprężenie
    sigma_max = 14000
    #Dopuszczalne ugięcie końca
    delta_max = 1e4
    penalty = 0.0
    lambda_pen = 1e6

    x = np.asarray(x)
    n = len(x)

    # Funkcja celu (objętość)
    f = np.sum(x)

    # Ograniczenia naprężeń
    i = np.arange(1, n + 1)
    sigma = 6 * P * (n - i + 1) * L / (x ** 2)
    stress_violation = np.maximum(0.0, sigma - sigma_max)

    # Ograniczenie ugięcia
    delta = (4 * P * L**3 / E) * np.sum((n - i + 1)**3 / x)
    delta_violation = max(0.0, delta - delta_max)

    penalty = lambda_pen * (
            np.sum(stress_violation ** 2) +
            delta_violation ** 2
    )

    return f + penalty

class MyFitnessFunctionWrapper(FitnessFunctionBase):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.function = self._wrapped_fitness_func

    def _wrapped_fitness_func(self, solution):
        return cantilever_objective(solution)

    def eval(self, params):
        return self.function(params)

def eng_de(method_id, dim):

    # lista metod
    methods = ["REINITIALIZE", "PROJECTION_LAMARCKIAN", "PROJECTION_DARWINIAN", "REFLECTION_LAMARCKIAN",
               "REFLECTION_DARWINIAN", "WRAPPING_LAMARCKIAN", "WRAPPING_DARWINIAN", "PROJECTION_MIDPOINT",
               "PENALTY_DEATH", "PENALTY_ADDITIVE", "PENALTY_SUBSTITUTION", "RAND_BASE", "MIDPOINT_BASE",
               "MIDPOINT_TARGET", "RESAMPLING", "CONSERVATIVE", "PROJECTION_BASE"]

    fitness_function = MyFitnessFunctionWrapper("Belka")

    params = DEData(
        population_size=100,
        dimension=dim,
        lb=[20.0]*dim,
        ub=[500.0]*dim,
        optimization_type=OptimizationType.MINIMIZATION,
        boundary_constraints_fun=getattr(BoundaryFixing, methods[method_id]),
        function=fitness_function,
        mutation_factor=0.5,
        crossover_rate=0.8,
        log_population=True,
        max_nfe=5000 * dim,
        show_plots=False
    )

    if dim <= 50:
        for i in range(0, 3):
            params.parallel_processing = ['thread', 20]
            default2 = DE(params, db_conn="Partial_data.db", db_auto_write=False)
            results = default2.run()
    else:
        params.parallel_processing = ['thread', 20]
        default2 = DE(params, db_conn="Partial_data.db", db_auto_write=False)
        results = default2.run()

def eng_opp(method_id, dim):

    # lista metod
    methods = ["REINITIALIZE", "PROJECTION_LAMARCKIAN", "PROJECTION_DARWINIAN", "REFLECTION_LAMARCKIAN",
               "REFLECTION_DARWINIAN", "WRAPPING_LAMARCKIAN", "WRAPPING_DARWINIAN", "PROJECTION_MIDPOINT",
               "PENALTY_DEATH", "PENALTY_ADDITIVE", "PENALTY_SUBSTITUTION", "RAND_BASE", "MIDPOINT_BASE",
               "MIDPOINT_TARGET", "RESAMPLING", "CONSERVATIVE", "PROJECTION_BASE"]

    fitness_function = MyFitnessFunctionWrapper("Belka")

    params = OppBasedData(
        population_size=100,
        dimension=dim,
        lb=[20.0] * dim,
        ub=[500.0] * dim,
        optimization_type=OptimizationType.MINIMIZATION,
        boundary_constraints_fun=getattr(BoundaryFixing, methods[method_id]),
        function=fitness_function,
        mutation_factor=0.5,
        crossover_rate=0.8,
        log_population=True,
        max_nfe=5000 * dim,
        show_plots=False
    )

    if dim <= 50:
        for i in range(0, 3):
            params.parallel_processing = ['thread', 20]
            default2 = OppBasedDE(params, db_conn="Partial_data.db", db_auto_write=False)
            results = default2.run()
    else:
        params.parallel_processing = ['thread', 20]
        default2 = OppBasedDE(params, db_conn="Partial_data.db", db_auto_write=False)
        results = default2.run()

def eng_ide(method_id, dim):

    # lista metod
    methods = ["REINITIALIZE", "PROJECTION_LAMARCKIAN", "PROJECTION_DARWINIAN", "REFLECTION_LAMARCKIAN",
               "REFLECTION_DARWINIAN", "WRAPPING_LAMARCKIAN", "WRAPPING_DARWINIAN", "PROJECTION_MIDPOINT",
               "PENALTY_DEATH", "PENALTY_ADDITIVE", "PENALTY_SUBSTITUTION", "RAND_BASE", "MIDPOINT_BASE",
               "MIDPOINT_TARGET", "RESAMPLING", "CONSERVATIVE", "PROJECTION_BASE"]

    fitness_function = MyFitnessFunctionWrapper("Belka")

    params = IDEData(
        population_size=100,
        dimension=dim,
        lb=[20.0] * dim,
        ub=[500.0] * dim,
        optimization_type=OptimizationType.MINIMIZATION,
        boundary_constraints_fun=getattr(BoundaryFixing, methods[method_id]),
        function=fitness_function,
        log_population=True,
        max_nfe=5000 * dim,
        show_plots=False
    )

    if dim <= 50:
        for i in range(0, 3):
            params.parallel_processing = ['thread', 20]
            default2 = IDE(params, db_conn="Partial_data.db", db_auto_write=False)
            results = default2.run()
    else:
        params.parallel_processing = ['thread', 20]
        default2 = IDE(params, db_conn="Partial_data.db", db_auto_write=False)
        results = default2.run()

def eng_eide(method_id, dim):

    # lista metod
    methods = ["REINITIALIZE", "PROJECTION_LAMARCKIAN", "PROJECTION_DARWINIAN", "REFLECTION_LAMARCKIAN",
               "REFLECTION_DARWINIAN", "WRAPPING_LAMARCKIAN", "WRAPPING_DARWINIAN", "PROJECTION_MIDPOINT",
               "PENALTY_DEATH", "PENALTY_ADDITIVE", "PENALTY_SUBSTITUTION", "RAND_BASE", "MIDPOINT_BASE",
               "MIDPOINT_TARGET", "RESAMPLING", "CONSERVATIVE", "PROJECTION_BASE"]

    fitness_function = MyFitnessFunctionWrapper("Belka")

    params = EIDEData(
        population_size=100,
        dimension=dim,
        lb=[20.0] * dim,
        ub=[500.0] * dim,
        optimization_type=OptimizationType.MINIMIZATION,
        boundary_constraints_fun=getattr(BoundaryFixing, methods[method_id]),
        function=fitness_function,
        crossover_rate_min=0.2,
        crossover_rate_max=0.8,
        log_population=True,
        max_nfe=5000 * dim,
        show_plots=False
    )

    if dim <= 50:
        for i in range(0, 3):
            params.parallel_processing = ['thread', 20]
            default2 = EIDE(params, db_conn="Partial_data.db", db_auto_write=False)
            results = default2.run()
    else:
        params.parallel_processing = ['thread', 20]
        default2 = EIDE(params, db_conn="Partial_data.db", db_auto_write=False)
        results = default2.run()

def test():
    '''

    a = [10, 20, 30, 50, 100]
    b = ['a', 'b', 'c']
    test = [(x, y) for x in a for y in b]

    a = [10, 20, 30, 50, 100]
    b = ['a', 'b', 'c']
    c = [11,12,13]

    for x,y,z in product(a,b,c):
        print(x,y,z)

    '''

    # lista wymiarów
    dim = [10]

    # lista funkcji
    t = list(range(1, 31))
    fun = [f"F{i}2014" for i in t]

    # lista metod
    methods = ["REINITIALIZE", "PROJECTION_LAMARCKIAN", "PROJECTION_DARWINIAN", "REFLECTION_LAMARCKIAN",
               "REFLECTION_DARWINIAN", "WRAPPING_LAMARCKIAN", "WRAPPING_DARWINIAN", "PROJECTION_MIDPOINT",
               "PENALTY_DEATH", "PENALTY_ADDITIVE", "PENALTY_SUBSTITUTION", "RAND_BASE", "MIDPOINT_BASE",
               "MIDPOINT_TARGET", "RESAMPLING", "CONSERVATIVE", "PROJECTION_BASE"]
    '''
    Done
    Funkcja:
    Metoda:
    '''

    # Indeks funkcji 0-29
    fun_ind = 0

    # indeks metody 0-16
    met_ind = 0

    for d in dim:
        fitness_fun_opf = FitnessFunctionOpfunu(
            func_type=getattr(opf, fun[fun_ind]),
            ndim=d
        )

        params = DEData(
            population_size=100,
            dimension=d,
            lb=[-100] * d,
            ub=[100] * d,
            optimization_type=OptimizationType.MINIMIZATION,
            boundary_constraints_fun=getattr(BoundaryFixing, methods[met_ind]),
            function=fitness_fun_opf,
            fun_optimum=fitness_fun_opf.function.f_global,
            fun_precision=1e-6,
            mutation_factor=0.5,
            crossover_rate=0.8,
            log_population=True,
            show_plots=False
        )

        for i in range(0, 5):
            params.parallel_processing = ['thread', 20]
            default2 = DE(params, db_conn="Segmented_data.db", db_auto_write=False)
            results = default2.run()

def test_eide():

    dim=100

    fitness_fun_opf = FitnessFunctionOpfunu(
        func_type=opf.F12014,
        ndim=dim
    )

    params = EIDEData(
        population_size=100,
        dimension=dim,
        lb=[-100] * dim,
        ub=[100] * dim,
        optimization_type=OptimizationType.MINIMIZATION,
        boundary_constraints_fun=BoundaryFixing.REINITIALIZE,
        function=fitness_fun_opf,
        crossover_rate_min=0.2,
        crossover_rate_max=0.8,
        log_population=True,
        max_nfe=10000 * dim,
        show_plots=False
    )


    params.parallel_processing = ['thread', 20]
    default2 = EIDE(params, db_conn="test.db", db_auto_write=False)
    results = default2.run()

def test_de(dim=10):

    print(dim)

    fitness_fun_opf = FitnessFunctionOpfunu(
        func_type=opf.F12014,
        ndim=dim
    )

    params = DEData(
        population_size=10,
        dimension=dim,
        lb=[-100] * dim,
        ub=[100] * dim,
        optimization_type=OptimizationType.MINIMIZATION,
        boundary_constraints_fun=BoundaryFixing.REINITIALIZE,
        function=fitness_fun_opf,
        mutation_factor=0.5,
        crossover_rate=0.8,
        log_population=True,
        show_plots=False,
        max_nfe=10000 * dim
    )

    params.parallel_processing = ['thread', 20]
    default2 = DE(params, db_conn="Partial_data.db", db_auto_write=False)
    results = default2.run()

def compute_ecdf(conn, alg_id, fun_id, method_id, dim, fun_opt):
    alg = ["DE", "OPP", "IDE", "EIDE"]
    fun = ["F12014", "F22014", "F32014", "F42014", "F52014", "Belka"]
    methods = ["REINITIALIZE", "PROJECTION_LAMARCKIAN", "PROJECTION_DARWINIAN", "REFLECTION_LAMARCKIAN",
               "REFLECTION_DARWINIAN", "WRAPPING_LAMARCKIAN", "WRAPPING_DARWINIAN", "PROJECTION_MIDPOINT",
               "PENALTY_DEATH", "PENALTY_ADDITIVE", "PENALTY_SUBSTITUTION", "RAND_BASE", "MIDPOINT_BASE",
               "MIDPOINT_TARGET", "RESAMPLING", "CONSERVATIVE", "PROJECTION_BASE"]

    queries = [
        f"SELECT * FROM {alg[alg_id]}_{fun[fun_id]}_{methods[method_id]}_dim{dim}_results_1",
        f"SELECT * FROM {alg[alg_id]}_{fun[fun_id]}_{methods[method_id]}_dim{dim}_results_2",
        f"SELECT * FROM {alg[alg_id]}_{fun[fun_id]}_{methods[method_id]}_dim{dim}_results_3"
    ]

    #15 1
    #10 7 5
    #30 50 9 7
    #100 10 8
    if fun_id == 0:
        if dim == 10:
            targets = 10.0 ** np.arange(7, 5, -1)
        elif dim == 30 or dim == 50:
            targets = 10.0 ** np.arange(9, 7, -1)
        elif dim == 100:
            targets = 10.0 ** np.arange(10, 8, -1)
    elif fun_id == 1:
        if dim == 10:
            targets = 10.0 ** np.arange(9, 7, -1)
        elif dim == 30 or dim == 50:
            targets = 10.0 ** np.arange(10, 8, -1)
        elif dim == 100:
            targets = 10.0 ** np.arange(11, 9, -1)
    elif fun_id == 2:
        if dim == 10:
            targets = 10.0 ** np.arange(4, 2, -1)
        elif dim == 30:
            targets = 10.0 ** np.arange(5, 3, -1)
        elif dim == 50:
            targets = 10.0 ** np.arange(6, 4, -1)
        elif dim == 100:
            targets = 10.0 ** np.arange(5.5, 4.5, -1)
    elif fun_id == 3:
        if dim == 10:
            targets = 10.0 ** np.arange(2, 0, -1)
        elif dim == 30:
            targets = 10.0 ** np.arange(4, 1, -1)
        elif dim == 50:
            targets = 10.0 ** np.arange(4, 2, -1)
        elif dim == 100:
            targets = 10.0 ** np.arange(4, 2, -1)
    elif fun_id == 4:
        if dim == 10:
            targets = 21.2 - np.arange(1, 0.1, -0.1)
        elif dim == 30:
            targets = 21.7 - np.arange(1, 0.1, -0.1)
        elif dim == 50:
            targets = 21.8 - np.arange(1, 0.1, -0.1)
        elif dim == 100:
            targets = 21.7 - np.arange(0.5, 0.1, -0.1)
    elif fun_id == 5:
        if dim == 10:
            targets = 10.0 ** np.arange(3, 1, -1)
        elif dim == 30:
            targets = 10.0 ** np.arange(4, 2, -1)
        elif dim == 50:
            targets = 10.0 ** np.arange(4, 2, -0.5)
        elif dim == 100:
            targets = 10.0 ** np.arange(5, 3, -1)

    all_y_targets = []

    for target_value in targets:
        all_x = []

        for query in queries:
            df = pd.read_sql_query(query, conn).copy()

            df['fes_per_dim'] = df['epoch'] / dim
            df['target_reached'] = abs(df['fitnessValueBest'] - fun_opt) <= target_value

            df_sorted = df.sort_values('fes_per_dim')

            x = df_sorted['fes_per_dim'].values
            y = np.cumsum(df_sorted['target_reached'].values) / len(df_sorted)

            all_x.append((x, y))

        # wspólna siatka X
        x_common = np.linspace(0, max(max(x) for x, _ in all_x), 500)

        ys_interp = []
        for x, y in all_x:
            y_interp = np.interp(x_common, x, y)
            ys_interp.append(y_interp)

        y_mean = np.mean(ys_interp, axis=0)
        all_y_targets.append(y_mean)

    # 🔥 średnia po targetach
    y_final = np.mean(all_y_targets, axis=0)

    # AUC
    auc = np.trapezoid(y_final, x_common)
    auc_norm = auc / (x_common[-1] - x_common[0])

    print("AUC dla", methods[method_id], "=", auc_norm)

    return x_common, y_final

def plot_ecdf(alg_id, fun_id, dim, set):
    methods = ["REINITIALIZE", "PROJECTION_LAMARCKIAN", "PROJECTION_DARWINIAN", "REFLECTION_LAMARCKIAN",
               "REFLECTION_DARWINIAN", "WRAPPING_LAMARCKIAN", "WRAPPING_DARWINIAN", "PROJECTION_MIDPOINT",
               "PENALTY_DEATH", "PENALTY_ADDITIVE", "PENALTY_SUBSTITUTION", "RAND_BASE", "MIDPOINT_BASE",
               "MIDPOINT_TARGET", "RESAMPLING", "CONSERVATIVE", "PROJECTION_BASE"]

    conn = sqlite3.connect("Partial_data.db")

    if fun_id == 0:
        fun_optimum = 100
    elif fun_id == 1:
        fun_optimum = 200
    elif fun_id == 2:
        fun_optimum = 300
    elif fun_id == 3:
        fun_optimum = 400
    elif fun_id == 4:
        fun_optimum = 500
    elif fun_id == 5:
        if dim == 10:
            fun_optimum = 332.26
        elif dim == 30:
            fun_optimum = 1200
        elif dim == 50:
            fun_optimum = 2000
        elif dim == 100:
            fun_optimum = 3300

    if set == 1:
        configs = [
            (0, "blue"),
            (1, "red"),
            (2, "green"),
            (3, "purple"),
            (4, "orange"),
        ]
    elif set == 2:
        configs = [
            (5, "blue"),
            (6, "red"),
            (7, "green"),
            (8, "purple"),
            (9, "orange"),
        ]
    elif set == 3:
        configs = [
            (10, "blue"),
            (11, "red"),
            (12, "green"),
            (13, "purple"),
            (14, "orange"),
        ]
    elif set == 4:
        configs = [
            (15, "blue"),
            (16, "red"),
        ]

    for method_id, color in configs:
        x, y = compute_ecdf(conn, alg_id, fun_id, method_id, dim, fun_optimum)
        plt.plot(x, y, label=methods[method_id], color=color)
    conn.close()

    plt.xlabel("FES / dim")
    plt.ylabel("ECDF")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(True)

    plt.show()
def plot():
    # połączenie z bazą
    conn = sqlite3.connect("Partial_data.db")
    # wczytanie całej tabeli do DataFrame
    df = pd.read_sql_query("SELECT * FROM EIDE_Belka_REINITIALIZE_dim10_results_1", conn)

    # zamknięcie połączenia
    conn.close()

    dim = 10
    fun_opt = 336.24
    max_fes = df['epoch'].max()

    targets = 10.0 ** np.arange(3, -9, -1)

    f_evals = []

    for target in targets:

        # błąd względem optimum
        error = abs(df['fitnessValueBest'] - fun_opt)

        reached = df[error <= target]

        if len(reached) > 0:
            fes = reached.iloc[0]['epoch']
        else:
            # target nieosiągnięty
            fes = max_fes

        f_evals.append(fes / dim)

    # sortowanie
    f_evals = np.sort(f_evals)

    # ECDF
    y = np.arange(1, len(f_evals) + 1) / len(f_evals)
    x = f_evals

    plt.figure(figsize=(10, 6))

    plt.step(x, y, where="post")

    plt.xscale("log")

    plt.xlabel("f-evals / dimension")
    plt.ylabel("Proportion of function + target pairs")

    plt.title("ECDF (BBOB-style)")

    plt.grid(True)
    plt.show()
def plot_exe():
    alg = ["DE", "OPP", "IDE", "EIDE"]
    fun = ["F12014", "F22014", "F32014", "F42014", "F52014", "Belka"]
    methods = ["REINITIALIZE", "PROJECTION_LAMARCKIAN", "PROJECTION_DARWINIAN", "REFLECTION_LAMARCKIAN",
               "REFLECTION_DARWINIAN", "WRAPPING_LAMARCKIAN", "WRAPPING_DARWINIAN", "PROJECTION_MIDPOINT",
               "PENALTY_DEATH", "PENALTY_ADDITIVE", "PENALTY_SUBSTITUTION", "RAND_BASE", "MIDPOINT_BASE",
               "MIDPOINT_TARGET", "RESAMPLING", "CONSERVATIVE", "PROJECTION_BASE"]

    query = f"SELECT * FROM {alg[3]}_{fun[5]}_{methods[0]}_dim10_results_1"

    # połączenie z bazą
    conn = sqlite3.connect("Partial_data.db")
    # wczytanie całej tabeli do DataFrame
    df = pd.read_sql_query(query, conn)

    # zamknięcie połączenia
    conn.close()

    print(df.head())

    # Ustal target fitness, np.
    target_value = 1e+3
    #Wymiar problemu
    dim = 10
    #Optimum funkcji
    fun_opt = 332.26

    # Normalizujemy FES przez wymiar problemu
    df['fes_per_dim'] = df['epoch'] / dim

    # Dla każdego rekordu sprawdzamy, czy target został osiągnięty
    df['target_reached'] = abs(df['fitnessValueBest'] - fun_opt) <= target_value

    # Sortujemy dane po FES/dimension
    df_sorted = df.sort_values('fes_per_dim')

    # Tworzymy ECDF: odsetek osiągniętych targetów w funkcji FES/dimension
    x = df_sorted['fes_per_dim'].values
    y = np.cumsum(df_sorted['target_reached'].values) / len(df_sorted)
    #y = np.arange(1, len(x)+1) / len(x)

    print(len(x))

    # Rysujemy wykres ECDF
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=2)
    #plt.step(x, y, where="post")
    #plt.xticks(np.arange(0, max(x) + 1, 100))
    #plt.xscale("log")
    plt.xlabel('F-Evals / Dimension')
    plt.ylabel('Proportion of targets reached')
    plt.title(f'ECDF of target fitness ≤ {target_value}')
    plt.grid(True)
    plt.show()

    '''
    fes_hits = []

    for run_id, g in df.groupby("run"):
        
        g = g.sort_values("epoch")
        
        hit = g[g["error"] <= target]
        
        if len(hit) > 0:
            fes_hits.append(hit.iloc[0]["epoch"] / dim)
            
    fes_hits = np.sort(fes_hits)

    x = fes_hits
    y = np.arange(1, len(x)+1) / len(x)
    '''

def all_plot():
    alg = ["DE", "OPP", "IDE", "EIDE"]
    fun = ["F12014", "F22014", "F32014", "F42014", "F52014", "Belka"]
    methods = ["REINITIALIZE", "PROJECTION_LAMARCKIAN", "PROJECTION_DARWINIAN", "REFLECTION_LAMARCKIAN",
               "REFLECTION_DARWINIAN", "WRAPPING_LAMARCKIAN", "WRAPPING_DARWINIAN", "PROJECTION_MIDPOINT",
               "PENALTY_DEATH", "PENALTY_ADDITIVE", "PENALTY_SUBSTITUTION", "RAND_BASE", "MIDPOINT_BASE",
               "MIDPOINT_TARGET", "RESAMPLING", "CONSERVATIVE", "PROJECTION_BASE"]

    query1 = f"SELECT * FROM {alg[3]}_{fun[5]}_{methods[0]}_dim10_results_1"
    query2 = f"SELECT * FROM {alg[3]}_{fun[5]}_{methods[0]}_dim10_results_2"
    query3 = f"SELECT * FROM {alg[3]}_{fun[5]}_{methods[0]}_dim10_results_3"

    # połączenie z bazą
    conn = sqlite3.connect("Partial_data.db")
    # wczytanie całej tabeli do DataFrame
    df1 = pd.read_sql_query(query1, conn)
    df2 = pd.read_sql_query(query2, conn)
    df3 = pd.read_sql_query(query3, conn)

    # zamknięcie połączenia
    conn.close()

    #print(df.head())

    # Ustal target fitness, np.
    target_value = 1e+2
    # Wymiar problemu
    dim = 10
    # Optimum funkcji
    fun_opt = 332.26

    all_x = []

    for df in [df1, df2, df3]:
        df = df.copy()
        df['fes_per_dim'] = df['epoch'] / dim
        df['target_reached'] = abs(df['fitnessValueBest'] - fun_opt) <= target_value

        df_sorted = df.sort_values('fes_per_dim')

        x = df_sorted['fes_per_dim'].values
        y = np.cumsum(df_sorted['target_reached'].values) / len(df_sorted)

        all_x.append((x, y))

    # wspólna siatka X
    x_common = np.linspace(0, max(max(x) for x, _ in all_x), 500)

    ys_interp = []
    for x, y in all_x:
        y_interp = np.interp(x_common, x, y)
        ys_interp.append(y_interp)

    y_mean = np.mean(ys_interp, axis=0)

    plt.plot(x_common, y_mean, label="Mean ECDF", linewidth=2)
    plt.legend()
    plt.show()
def plot_2():
    # połączenie z bazą
    conn = sqlite3.connect("Partial_data.db")
    # wczytanie całej tabeli do DataFrame
    run1 = pd.read_sql_query("SELECT * FROM DE_F12014_REINITIALIZE_dim10_results_1", conn)
    run2 = pd.read_sql_query("SELECT * FROM DE_F12014_REINITIALIZE_dim10_results_2", conn)
    run3 = pd.read_sql_query("SELECT * FROM DE_F12014_REINITIALIZE_dim10_results_3", conn)
    run4 = pd.read_sql_query("SELECT * FROM DE_F12014_REINITIALIZE_dim10_results_4", conn)
    run5 = pd.read_sql_query("SELECT * FROM DE_F12014_REINITIALIZE_dim10_results_5", conn)

    # zamknięcie połączenia
    conn.close()

    runs = [run1, run2, run3, run4, run5]

    dimension = 10
    max_evals = 100000

    targets = [1e3, 1e2, 1e1, 1e0, 1e-1, 1e-2, 1e-3]

    f_evals = []

    for df in runs:
        for t in targets:
            hit = df[df["fitnessValueBest"] <= t]

            if len(hit) > 0:
                evals = hit.iloc[0]["epoch"]
            else:
                evals = max_evals

            f_evals.append(evals / dimension)

    f_evals = np.array(sorted(f_evals))

    ecdf_y = np.arange(1, len(f_evals) + 1) / len(f_evals)

    plt.step(f_evals, ecdf_y, where="post")

    plt.xscale("log")

    plt.xlabel("f-evals / dimension")
    plt.ylabel("Proportion of function + target pairs")

    plt.grid(True)
    plt.show()


def exe_de(fun_id, method_id, dim):

    # lista funkcji
    t = list(range(1, 31))
    fun = [f"F{i}2014" for i in t]

    # lista metod
    methods = ["REINITIALIZE", "PROJECTION_LAMARCKIAN", "PROJECTION_DARWINIAN", "REFLECTION_LAMARCKIAN",
               "REFLECTION_DARWINIAN", "WRAPPING_LAMARCKIAN", "WRAPPING_DARWINIAN", "PROJECTION_MIDPOINT",
               "PENALTY_DEATH", "PENALTY_ADDITIVE", "PENALTY_SUBSTITUTION", "RAND_BASE", "MIDPOINT_BASE",
               "MIDPOINT_TARGET", "RESAMPLING", "CONSERVATIVE", "PROJECTION_BASE"]

    #for fun,method,dim in product(fun,methods,dim):

    fitness_fun_opf = FitnessFunctionOpfunu(
        func_type=getattr(opf, fun[fun_id]),
        ndim=dim
    )

    params = DEData(
        population_size=100,
        dimension=dim,
        lb=[-100] * dim,
        ub=[100] * dim,
        optimization_type=OptimizationType.MINIMIZATION,
        boundary_constraints_fun=getattr(BoundaryFixing, methods[method_id]),
        function=fitness_fun_opf,
        mutation_factor=0.5,
        crossover_rate=0.8,
        log_population=True,
        show_plots=False,
        max_nfe=5000 * dim
    )

    if dim <= 50:
        for i in range(0, 3):
            params.parallel_processing = ['thread', 20]
            default2 = DE(params, db_conn="Partial_data.db", db_auto_write=False)
            results = default2.run()
    else:
        params.parallel_processing = ['thread', 20]
        default2 = DE(params, db_conn="Partial_data.db", db_auto_write=False)
        results = default2.run()

def exe_opp(fun_id, method_id, dim):

    # lista funkcji
    t = list(range(1, 31))
    fun = [f"F{i}2014" for i in t]

    # lista metod
    methods = ["REINITIALIZE", "PROJECTION_LAMARCKIAN", "PROJECTION_DARWINIAN", "REFLECTION_LAMARCKIAN",
               "REFLECTION_DARWINIAN", "WRAPPING_LAMARCKIAN", "WRAPPING_DARWINIAN", "PROJECTION_MIDPOINT",
               "PENALTY_DEATH", "PENALTY_ADDITIVE", "PENALTY_SUBSTITUTION", "RAND_BASE", "MIDPOINT_BASE",
               "MIDPOINT_TARGET", "RESAMPLING", "CONSERVATIVE", "PROJECTION_BASE"]

    fitness_fun_opf = FitnessFunctionOpfunu(
        func_type=getattr(opf, fun[fun_id]),
        ndim=dim
    )

    params = OppBasedData(
        population_size=100,
        dimension=dim,
        lb=[-100]*dim,
        ub=[100]*dim,
        optimization_type=OptimizationType.MINIMIZATION,
        boundary_constraints_fun=getattr(BoundaryFixing, methods[method_id]),
        function=fitness_fun_opf,
        mutation_factor=0.5,
        crossover_rate=0.8,
        log_population=True,
        max_nfe=5000*dim,
        show_plots=False
    )

    if dim <= 50:
        for i in range(0, 3):
            params.parallel_processing = ['thread', 20]
            default2 = OppBasedDE(params, db_conn="Partial_data.db", db_auto_write=False)
            results = default2.run()
    else:
        params.parallel_processing = ['thread', 20]
        default2 = OppBasedDE(params, db_conn="Partial_data.db", db_auto_write=False)
        results = default2.run()

def exe_ide(fun_id, method_id, dim):

    # lista funkcji
    t = list(range(1, 31))
    fun = [f"F{i}2014" for i in t]

    # lista metod
    methods = ["REINITIALIZE", "PROJECTION_LAMARCKIAN", "PROJECTION_DARWINIAN", "REFLECTION_LAMARCKIAN",
               "REFLECTION_DARWINIAN", "WRAPPING_LAMARCKIAN", "WRAPPING_DARWINIAN", "PROJECTION_MIDPOINT",
               "PENALTY_DEATH", "PENALTY_ADDITIVE", "PENALTY_SUBSTITUTION", "RAND_BASE", "MIDPOINT_BASE",
               "MIDPOINT_TARGET", "RESAMPLING", "CONSERVATIVE", "PROJECTION_BASE"]

    fitness_fun_opf = FitnessFunctionOpfunu(
        func_type=getattr(opf, fun[fun_id]),
        ndim=dim
    )

    params = IDEData(
        population_size=100,
        dimension=dim,
        lb=[-100]*dim,
        ub=[100]*dim,
        optimization_type=OptimizationType.MINIMIZATION,
        boundary_constraints_fun=getattr(BoundaryFixing, methods[method_id]),
        function=fitness_fun_opf,
        log_population=True,
        max_nfe=5000 * dim,
        show_plots=False
    )

    if dim <= 50:
        for i in range(0, 3):
            params.parallel_processing = ['thread', 20]
            default2 = IDE(params, db_conn="Partial_data.db", db_auto_write=False)
            results = default2.run()
    else:
        params.parallel_processing = ['thread', 20]
        default2 = IDE(params, db_conn="Partial_data.db", db_auto_write=False)
        results = default2.run()

def exe_eide(fun_id, method_id, dim):

    # lista funkcji
    t = list(range(1, 31))
    fun = [f"F{i}2014" for i in t]

    # lista metod
    methods = ["REINITIALIZE", "PROJECTION_LAMARCKIAN", "PROJECTION_DARWINIAN", "REFLECTION_LAMARCKIAN",
               "REFLECTION_DARWINIAN", "WRAPPING_LAMARCKIAN", "WRAPPING_DARWINIAN", "PROJECTION_MIDPOINT",
               "PENALTY_DEATH", "PENALTY_ADDITIVE", "PENALTY_SUBSTITUTION", "RAND_BASE", "MIDPOINT_BASE",
               "MIDPOINT_TARGET", "RESAMPLING", "CONSERVATIVE", "PROJECTION_BASE"]

    fitness_fun_opf = FitnessFunctionOpfunu(
        func_type=getattr(opf, fun[fun_id]),
        ndim=dim
    )

    params = EIDEData(
        population_size=100,
        dimension=dim,
        lb=[-100]*dim,
        ub=[100]*dim,
        optimization_type=OptimizationType.MINIMIZATION,
        boundary_constraints_fun=getattr(BoundaryFixing, methods[method_id]),
        function=fitness_fun_opf,
        crossover_rate_min=0.2,
        crossover_rate_max=0.8,
        log_population=True,
        max_nfe=5000 * dim,
        show_plots=False
    )

    if dim <= 50:
        for i in range(0, 3):
            params.parallel_processing = ['thread', 20]
            default2 = EIDE(params, db_conn="Partial_data.db", db_auto_write=False)
            results = default2.run()
    else:
        params.parallel_processing = ['thread', 20]
        default2 = EIDE(params, db_conn="Partial_data.db", db_auto_write=False)
        results = default2.run()

def db_data():
    db_conn = "Partial_data.db"
    db = SQLiteConnector(db_conn)
    db.connect()

    db.execute_query('DROP TABLE DE_F12014_REINITIALIZE_dim50_results_2')

    db.close()


if __name__ == "__main__":

    #dim = int(sys.argv[2])
    #method_id = int(sys.argv[1])
    fun_id = 4

    #plot_exe()

    '''
    0 - DE, 1 - OPP, 2 - IDE, 3 - EIDE
    Belka: 10 - 332.26, 30 - 976, 50 - 1625, 100 - 3300
            1e2         1200 1e3  2000 2e3   1e4
    AUC: 
    '''

    plot_ecdf(3, 5, 50, 3)
    #plot_ecdf(3, 5, 10, 1)

    #exe_eide(fun_id, method_id, dim)
    #exe_eide(0, 1, 100)
    #eng_eide(method_id, dim)

    #exe_ide(fun_id, method_id, dim)
    #eng_ide(method_id, dim)

    #exe_opp(fun_id, method_id, dim)
    #eng_opp(method_id, dim)

    #exe_de(fun_id, method_id, dim)
    #eng_de(method_id, dim)

    #0,1,3,5,7,8,9,10,11,12,13,14,15,16
    #9

    #test_de(dim)
    #db_data()
    #plot_exe()








