from detpy.DETAlgs.data.alg_data import MGDEData, EIDEData
from detpy.DETAlgs.eide import EIDE
from detpy.DETAlgs.mgde import MGDE
from detpy.models.enums.basevectorschema import BaseVectorSchema
from detpy.models.enums.boundary_constrain import BoundaryFixing
from detpy.models.enums.optimization import OptimizationType
from detpy.models.fitness_function import FitnessFunctionBase
import numpy as np
def cantilever_objective(x):
    # Dane
    P = 50000
    E = 2e7
    L = 100
    sigma_max = 14000
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
    delta = (P / E) * np.sum(((np.arange(1,6)*L)**3) / x)
    if delta > delta_max:
        penalty += (delta - delta_max)**2

    return f + lambda_pen * penalty

class MyFitnessFunctionWrapper(FitnessFunctionBase):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.function = self._wrapped_fitness_func

    def _wrapped_fitness_func(self, solution):
        return MyFitnessFunc(solution)

    def eval(self, params):
        return self.function(params)


def MyFitnessFunc(solution):
    print("sol", solution)
    return solution[0] + solution[1]


if __name__ == "__main__":
    a = [0, 5]
    b = [5, 10]
    fitness_function = MyFitnessFunctionWrapper("Function name")

    params = EIDEData(
        max_nfe=1000,
        population_size=10,
        dimension=len(a),
        lb=a,
        ub=b,
        optimization_type=OptimizationType.MINIMIZATION,
        base_vector_schema=BaseVectorSchema.CURRENT,
        boundary_constraints_fun=BoundaryFixing.RAND_BASE,
        function=fitness_function,
        log_population=True,
        y=3
    )
    params.parallel_processing = ['thread', 5]

    default2 = EIDE(params, db_conn="Differential_evolution.db", db_auto_write=False)
    results = default2.run()
    default2.write_results_to_database(results.epoch_metrics)
