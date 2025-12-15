import opfunu.cec_based.cec2014 as opf

from detpy.DETAlgs.data.alg_data import DEData
from detpy.DETAlgs.de import DE
from detpy.models.enums.boundary_constrain import BoundaryFixing
from detpy.models.enums.optimization import OptimizationType
from detpy.models.fitness_function import FitnessFunctionOpfunu

if __name__ == "__main__":
    num_of_epochs = 10

    fitness_fun_opf = FitnessFunctionOpfunu(
        func_type=opf.F82014,
        ndim=10
    )

    func = opf.F82014(ndim=10)
    print(func.f_global)
    print(func.x_global)

    params = DEData(
        population_size=10,
        dimension=10,
        lb=[-5,-100,-100,-100,-100,-100,-100,-100,-100,-100],
        ub=[5,100,100,100,100,100,100,100,100,100],
        optimization_type=OptimizationType.MINIMIZATION,
        boundary_constraints_fun=BoundaryFixing.PENALTY_ADDITIVE,
        function=fitness_fun_opf,
        mutation_factor=0.5,
        crossover_rate=0.8,
        log_population=True
    )
    params.parallel_processing = ['thread', 5]
    default2 = DE(params, db_conn="Differential_evolution.db", db_auto_write=False)
    results = default2.run()









