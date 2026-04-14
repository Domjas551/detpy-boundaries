from detpy.DETAlgs.base import BaseAlg
from detpy.DETAlgs.data.alg_data import OppBasedData
from detpy.DETAlgs.methods.methods_opposition_based import opp_based_generation_jumping
from detpy.DETAlgs.methods.methods_de import mutation, selection, crossing
from detpy.models.enums.boundary_constrain import BoundaryFixing, fix_boundary_constraints_full


class OppBasedDE(BaseAlg):
    """
        OppBasedDE

        Links:
        https://ieeexplore.ieee.org/document/4358759

        References:
        S. Rahnamayan, H. R. Tizhoosh and M. M. A. Salama, "Opposition-Based Differential Evolution,"
        in IEEE Transactions on Evolutionary Computation, vol. 12, no. 1, pp. 64-79, Feb. 2008,
        doi: 10.1109/TEVC.2007.894200.
    """

    def __init__(self, params: OppBasedData, db_conn=None, db_auto_write=False):
        super().__init__(OppBasedDE.__name__, params, db_conn, db_auto_write)

        self.mutation_factor = params.mutation_factor  # F
        self.crossover_rate = params.crossover_rate  # Cr
        self.crossing_type = params.crossing_type
        self.y = params.y
        self.base_vector_schema = params.base_vector_schema
        self.nfc = 0  # number of function calls
        self.max_nfc = params.max_nfc
        self.jumping_rate = params.jumping_rate

    def next_epoch(self):
        # New population after mutation
        v_pop = mutation(self._pop, base_vector_schema=self.base_vector_schema,
                         optimization_type=self.optimization_type, y=self.y, f=self.mutation_factor)

        # Apply boundary constrains on population in place
        # TODO corrected code boundary full
        if self.boundary_constraints_fun not in (BoundaryFixing.PROJECTION_DARWINIAN, BoundaryFixing.REFLECTION_DARWINIAN,
                                                 BoundaryFixing.WRAPPING_DARWINIAN, BoundaryFixing.PENALTY_DEATH,
                                                 BoundaryFixing.PENALTY_ADDITIVE, BoundaryFixing.PENALTY_SUBSTITUTION):
            fix_boundary_constraints_full(self._pop, v_pop, self._function.eval, self.base_vector_schema,
                                          self.optimization_type, self.y, self.mutation_factor,
                                          self.boundary_constraints_fun)

        # New population after crossing
        u_pop = crossing(self._pop, v_pop, cr=self.crossover_rate, crossing_type=self.crossing_type)

        # Update values before selection
        u_pop.update_fitness_values(self._function.eval, self.parallel_processing)
        self.nfc += self.population_size

        # Methods with Darwinian repair should be used before selection
        if self.boundary_constraints_fun in (BoundaryFixing.PROJECTION_DARWINIAN, BoundaryFixing.REFLECTION_DARWINIAN,
                                             BoundaryFixing.WRAPPING_DARWINIAN, BoundaryFixing.PENALTY_DEATH,
                                             BoundaryFixing.PENALTY_ADDITIVE, BoundaryFixing.PENALTY_SUBSTITUTION):
            fix_boundary_constraints_full(self._pop, u_pop, self._function.eval, self.base_vector_schema,
                                          self.optimization_type, self.y, self.mutation_factor,
                                          self.boundary_constraints_fun)

        # Select new population
        new_pop = selection(self._pop, u_pop)

        # Generation jumping
        if opp_based_generation_jumping(new_pop, self.jumping_rate, self._function.eval, self.parallel_processing):
            self.nfc += self.population_size

        # Override data
        self._pop = new_pop

