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
        lb=[-100] * dim,
        ub=[100] * dim,
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
        lb=[-100] * dim,
        ub=[100] * dim,
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
        lb=[-100] * dim,
        ub=[100] * dim,
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

def plot():
    # połączenie z bazą
    conn = sqlite3.connect("Partial_data.db")
    # wczytanie całej tabeli do DataFrame
    df = pd.read_sql_query("SELECT * FROM DE_F12014_REINITIALIZE_dim10_results_1", conn)

    # zamknięcie połączenia
    conn.close()

    dim = 10
    fun_opt = 100
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
    # połączenie z bazą
    conn = sqlite3.connect("Partial_data.db")
    # wczytanie całej tabeli do DataFrame
    df = pd.read_sql_query("SELECT * FROM DE_F12014_REINITIALIZE_dim10_results_1", conn)

    # zamknięcie połączenia
    conn.close()

    print(df.head())

    # Ustal target fitness, np.
    target_value = 1e+3
    #Wymiar problemu
    dim = 10
    #Optimum funkcji
    fun_opt = 100

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
        boundary_constraints_fun=getattr(BoundaryFixing,methods[method_id]),
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
        func_type=getattr(opf, fun_id),
        ndim=dim
    )

    params = IDEData(
        population_size=100,
        dimension=dim,
        lb=[-100]*dim,
        ub=[100]*dim,
        optimization_type=OptimizationType.MINIMIZATION,
        boundary_constraints_fun=getattr(BoundaryFixing,methods[method_id]),
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
        func_type=getattr(opf,fun_id),
        ndim=dim
    )

    params = EIDEData(
        population_size=100,
        dimension=dim,
        lb=[-100]*dim,
        ub=[100]*dim,
        optimization_type=OptimizationType.MINIMIZATION,
        boundary_constraints_fun=getattr(BoundaryFixing,methods[method_id]),
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

    #exe_de(fun_id, 7, 30)
    eng_de(16, 30)

    #0,1,3,5,7,8,9,10,11,12,13,14,15,16
    '''
    exe_de()
    exe_opp()
    exe_ide()
    exe_eide()
    '''
    #test_de(dim)
    #db_data()
    #plot_exe()








