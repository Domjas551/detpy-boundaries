import copy
from enum import Enum
import numpy as np
import random

from detpy.DETAlgs.methods.methods_de import get_best_member, mutation_ind
from detpy.models.enums.basevectorschema import BaseVectorSchema
from detpy.models.member import Member
from detpy.models.population import Population


class BoundaryFixing(Enum):
    REINITIALIZE = 'reinitialize'
    PROJECTION_LAMARCKIAN = 'projection_lamarckian'
    PROJECTION_DARWINIAN = 'projection_darwinian'
    REFLECTION_LAMARCKIAN = 'reflection_lamarckian'
    REFLECTION_DARWINIAN = 'reflection_darwinian'
    WRAPPING_LAMARCKIAN = 'wrapping_lamarckian'
    WRAPPING_DARWINIAN = 'wrapping_darwinian'
    REFLECTION_MIDPOINT = 'reflection_midpoint'
    PENALTY_DEATH = 'penalty_death'
    PENALTY_ADDITIVE = 'penalty_additive'
    PENALTY_SUBSTITUTION = 'penalty_substitution'
    RAND_BASE = 'rand_base'
    MIDPOINT_BASE = 'midpoint_base'
    MIDPOINT_TARGET = 'midpoint_target'
    RESAMPLING = 'resampling'
    CONSERVATIVE = 'conservative'
    PROJECTION_BASE = 'projection_base'
    REFLECTION_BACK = 'reflection_back'


def get_boundary_constraints_fun(fix_type: BoundaryFixing):
    return {
        BoundaryFixing.REINITIALIZE: lambda member: boundary_reinitialization(member),
        BoundaryFixing.PROJECTION_LAMARCKIAN: lambda member, fitness_fun: boundary_projection(member, fitness_fun, 1),
        BoundaryFixing.PROJECTION_DARWINIAN: lambda member, fitness_fun: boundary_projection(member, fitness_fun, 2),
        BoundaryFixing.REFLECTION_LAMARCKIAN: lambda member, fitness_fun: boundary_reflection(member, fitness_fun, 1),
        BoundaryFixing.REFLECTION_DARWINIAN: lambda member, fitness_fun: boundary_reflection(member, fitness_fun, 2),
        BoundaryFixing.WRAPPING_LAMARCKIAN: lambda member, fitness_fun: boundary_wrapping(member, fitness_fun, 1),
        BoundaryFixing.WRAPPING_DARWINIAN: lambda member, fitness_fun: boundary_wrapping(member, fitness_fun, 2),
        BoundaryFixing.REFLECTION_MIDPOINT: lambda member: boundary_projection_to_midpoint(member),
        BoundaryFixing.PENALTY_DEATH: lambda member: boundary_penalty_death(member),
        BoundaryFixing.PENALTY_ADDITIVE: lambda member, fitness_fun: boundary_penalty_additive(member, fitness_fun),
        BoundaryFixing.PENALTY_SUBSTITUTION: lambda member: boundary_penalty_substitution(member),
        BoundaryFixing.RAND_BASE: lambda trial, base_vector: boundary_rand_base(trial, base_vector),
        BoundaryFixing.MIDPOINT_BASE: lambda trial, base_vector: boundary_midpoint_base(trial, base_vector),
        BoundaryFixing.MIDPOINT_TARGET: lambda trial, parent: boundary_midpoint_target(trial, parent),
        BoundaryFixing.RESAMPLING: lambda population_trial, population, fitness_fun,
                                          base_vector_schema, optimization_type,
                                          mutation_y, mutation_f: boundary_resampling(population_trial, population,
                                                                                      fitness_fun, base_vector_schema,
                                                                                      optimization_type, mutation_y,
                                                                                      mutation_f),
        BoundaryFixing.CONSERVATIVE: lambda trial, base_vector: boundary_conservative(trial, base_vector),
        BoundaryFixing.PROJECTION_BASE: lambda trial, base_vector: boundary_projection_to_base(trial, base_vector),
        BoundaryFixing.REFLECTION_BACK: lambda member: boundary_reflection_back(member),
    }.get(fix_type, lambda: None)


def fix_boundary_constraints(population: Population, fix_type: BoundaryFixing):
    boundary_constraints_fun = get_boundary_constraints_fun(fix_type)
    if fix_type == BoundaryFixing.MIDPOINT_BASE:
        raise ValueError("fix_type=WITH_PARENT requires a trial population.")
    else:
        for member in population.members:
            if not member.is_member_in_interval():
                boundary_constraints_fun(member)


def fix_boundary_constraints_full(population: Population, trial: Population, fitness_fun, base_vector_schema,
                                         optimization_type, mutation_y, mutation_f, fix_type: BoundaryFixing):
    boundary_constraints_fun = get_boundary_constraints_fun(fix_type)

    #Methods requiring base vector
    if fix_type in (BoundaryFixing.MIDPOINT_BASE, BoundaryFixing.RAND_BASE,
                    BoundaryFixing.CONSERVATIVE, BoundaryFixing.PROJECTION_BASE):
        #Depending on base vector schema base vector is chosen
        if base_vector_schema == BaseVectorSchema.RAND:
            base_vector = random.choice(population.members.tolist())
            for member_trial in trial.members:
                boundary_constraints_fun(member_trial, base_vector)
        elif base_vector_schema == BaseVectorSchema.CURRENT:
            for member_trial, member_parent in zip(trial.members, population.members):
                boundary_constraints_fun(member_trial, member_parent)
        elif base_vector_schema == BaseVectorSchema.BEST:
            best_member = get_best_member(optimization_type, population)
            base_vector = best_member
            for member_trial in trial.members:
                boundary_constraints_fun(member_trial, base_vector)
    #Method requiring parent member
    elif fix_type == BoundaryFixing.MIDPOINT_TARGET:
        for member_trial, member_parent in zip(trial.members, population.members):
            boundary_constraints_fun(member_trial, member_parent)
    #Method requring redoing of a mutation
    elif fix_type == BoundaryFixing.RESAMPLING:
        boundary_constraints_fun(trial, population, fitness_fun, base_vector_schema, optimization_type, mutation_y, mutation_f)
    #Methods requiring fitness function
    elif fix_type in (BoundaryFixing.PENALTY_ADDITIVE, BoundaryFixing.PROJECTION_LAMARCKIAN,
                      BoundaryFixing.PROJECTION_DARWINIAN, BoundaryFixing.REFLECTION_LAMARCKIAN,
                      BoundaryFixing.REFLECTION_DARWINIAN, BoundaryFixing.WRAPPING_LAMARCKIAN,
                      BoundaryFixing.WRAPPING_DARWINIAN):
        for member in trial.members:
            if not member.is_member_in_interval():
                boundary_constraints_fun(member, fitness_fun)
    #Other methods
    else:
        for member in trial.members:
            if not member.is_member_in_interval():
                boundary_constraints_fun(member)


# Strategies for fixing members, when they are beyond boundaries

#Fixing strategies
def boundary_reinitialization(member: Member):
    """
    Reinitializes within given bound constraints.

    Args:
        member: member to be reinitialized
    """
    for chromosome in member.chromosomes:
        if chromosome.real_value > chromosome.ub or chromosome.real_value < chromosome.lb:
            chromosome.real_value = np.random.uniform(chromosome.lb, chromosome.ub)


def boundary_projection(member: Member, fitness_fun, type: int):
    """
    Projects violated constrain into violated constrain value.

    Args:
        member: member to be projected
        fitness_fun: fitness function
        type: 1 - Lamarckian repair, 2 - Darwinian repair
    """
    if type == 1:
        for chromosome in member.chromosomes:
            if chromosome.real_value > chromosome.ub:
                chromosome.real_value = chromosome.ub
            elif chromosome.real_value < chromosome.lb:
                chromosome.real_value = chromosome.lb
    elif type == 2:
        member_repaired = copy.deepcopy(member)
        for chromosome in member_repaired.chromosomes:
            if chromosome.real_value > chromosome.ub:
                chromosome.real_value = chromosome.ub
            elif chromosome.real_value < chromosome.lb:
                chromosome.real_value = chromosome.lb
        member_repaired.calculate_fitness_fun(fitness_fun)
        member.fitness_value = member_repaired.fitness_value


def boundary_reflection(member: Member, fitness_fun, type: int):
    """
    After repair new values can also be infeasible, to avoid this we repeat the process until we get feasible ones.

    Args:
        member: member to be reflected
        fitness_fun: fitness function
        type: 1 - Lamarckian repair, 2 - Darwinian repair
    """
    if type == 1:
        for chromosome in member.chromosomes:
            while True:
                if chromosome.real_value > chromosome.ub:
                    chromosome.real_value = 2 * chromosome.ub - chromosome.real_value
                elif chromosome.real_value < chromosome.lb:
                    chromosome.real_value = 2 * chromosome.lb - chromosome.real_value
                if chromosome.ub > chromosome.real_value > chromosome.lb:
                    break
    elif type == 2:
        member_repaired = copy.deepcopy(member)
        for chromosome in member_repaired.chromosomes:
            while True:
                if chromosome.real_value > chromosome.ub:
                    chromosome.real_value = 2 * chromosome.ub - chromosome.real_value
                elif chromosome.real_value < chromosome.lb:
                    chromosome.real_value = 2 * chromosome.lb - chromosome.real_value
                if chromosome.ub > chromosome.real_value > chromosome.lb:
                    break
        member_repaired.calculate_fitness_fun(fitness_fun)
        member.fitness_value = member_repaired.fitness_value


def boundary_wrapping(member: Member, fitness_fun, type: int):
    """
    After repair new values can also be infeasible, to avoid this we repeat the process until we get feasible ones.

    Args:
        member: member to be wrapped
        fitness_fun: fitness function
        type: 1 - Lamarckian repair, 2 - Darwinian repair
    """
    if type == 1:
        for chromosome in member.chromosomes:
            while True:
                if chromosome.real_value < chromosome.lb:
                    chromosome.real_value = chromosome.ub + (chromosome.real_value - chromosome.lb)
                elif chromosome.real_value > chromosome.ub:
                    chromosome.real_value = chromosome.lb + (chromosome.real_value - chromosome.ub)

                if chromosome.ub > chromosome.real_value > chromosome.lb:
                    break
    elif type == 2:
        member_repaired = copy.deepcopy(member)
        for chromosome in member_repaired.chromosomes:
            while True:
                if chromosome.real_value < chromosome.lb:
                    chromosome.real_value = chromosome.ub + (chromosome.real_value - chromosome.lb)
                elif chromosome.real_value > chromosome.ub:
                    chromosome.real_value = chromosome.lb + (chromosome.real_value - chromosome.ub)

                if chromosome.ub > chromosome.real_value > chromosome.lb:
                    break
        member_repaired.calculate_fitness_fun(fitness_fun)
        member.fitness_value = member_repaired.fitness_value


def boundary_projection_to_midpoint(member: Member):
    """
    Args:
        member: member to be projected
    """

    x = np.array([ch.real_value for ch in member.chromosomes], dtype=float)
    lb = np.array([ch.lb for ch in member.chromosomes], dtype=float)
    ub = np.array([ch.ub for ch in member.chromosomes], dtype=float)

    midpoint = (lb + ub) / 2
    direction = x - midpoint

    if np.all((x < lb) & (x > ub)):

        alphas=[]

        #Calculation of alpha
        for j in range(len(x)):
            d = direction[j]
            if d == 0:
                continue

            alpha_l = (lb[j] - midpoint[j]) / d
            alpha_u = (ub[j] - midpoint[j]) / d

            alpha_max_j = max(alpha_l, alpha_u)

            if alpha_max_j >= 0:
                alphas.append(alpha_max_j)


        if len(alphas) == 0:
            alpha = 0.0
        else:
            alpha = min(1.0, max(0.0, min(alphas)))

        r = midpoint + alpha * direction

        for j, ch in enumerate(member.chromosomes):
            ch.real_value = r[j]


#Penalty strategies
def boundary_penalty_death(member: Member):
    """
    Args:
        member: member to be sentenced
    """

    q = 1e12
    is_not_valid = False
    #We check if any of member's chromosome is outside of boundaries
    for chromosome in member.chromosomes:
        if chromosome.real_value > chromosome.ub or chromosome.real_value < chromosome.lb:
            is_not_valid = True
            break

    if is_not_valid:
        member.fitness_value = q


def boundary_penalty_additive(member: Member, fitness_fun):
    """
    Args:
        member: member to be sentenced
        fitness_fun: fitness function
    """

    is_not_valid = False
    # We check if any of member's chromosome is outside of boundaries
    for chromosome in member.chromosomes:
        if chromosome.real_value > chromosome.ub or chromosome.real_value < chromosome.lb:
            is_not_valid = True
            break

    if is_not_valid:
        member_fixed = copy.deepcopy(member)
        boundary_projection(member_fixed, fitness_fun, 1)
        member_fixed.calculate_fitness_fun(fitness_fun)
        alfa = 10
        penalty = 0.0

        for chromosome in member.chromosomes:
            if chromosome.real_value > chromosome.ub:
                penalty += (chromosome.real_value - chromosome.ub)**2
            elif chromosome.real_value < chromosome.lb:
                penalty += (chromosome.lb - chromosome.real_value)**2

        member.fitness_value = member_fixed.fitness_value + alfa * penalty


def boundary_penalty_substitution(member: Member):
    """
    Args:
        member: member to be sentenced
    """

    q = 1e12
    is_not_valid = False
    # We check if any of member's chromosome is outside of boundaries
    for chromosome in member.chromosomes:
        if chromosome.real_value > chromosome.ub or chromosome.real_value < chromosome.lb:
            is_not_valid = True
            break

    if is_not_valid:
        penalty = 0.0

        for chromosome in member.chromosomes:
            if chromosome.real_value > chromosome.ub:
                penalty += (chromosome.real_value - chromosome.ub)**2
            elif chromosome.real_value < chromosome.lb:
                penalty += (chromosome.lb - chromosome.real_value)**2

        member.fitness_value = q + penalty


#Feasibility preservation strategies
def boundary_rand_base(member_trial: Member, base_vector: Member):
    """
        Modifies the values 'member' relative to its base vector.

        :param member_trial: The member to be modified.
        :param base_vector: The base vector used as a reference.
    """
    for chromosome_trial, chromosome_vector in zip(member_trial.chromosomes, base_vector.chromosomes):
        if chromosome_trial.real_value < chromosome_trial.lb:
            chromosome_trial.real_value = np.random.uniform(chromosome_trial.lb, chromosome_vector.real_value)
        elif chromosome_trial.real_value > chromosome_trial.ub:
            chromosome_trial.real_value = np.random.uniform(chromosome_vector.real_value, chromosome_trial.ub)


def boundary_midpoint_base(member_trial: Member, base_vector: Member):
    """
    Modifies the values 'member' relative to its base vector.

    :param member_trial: The member to be modified.
    :param base_vector: The base vector used as a reference.
    """
    for chromosome_trial, chromosome_vector in zip(member_trial.chromosomes, base_vector.chromosomes):
        if chromosome_trial.real_value > chromosome_trial.ub:
            chromosome_trial.real_value = (chromosome_trial.ub + chromosome_vector.real_value) / 2
        elif chromosome_trial.real_value < chromosome_trial.lb:
            chromosome_trial.real_value = (chromosome_trial.lb + chromosome_vector.real_value) / 2


def boundary_midpoint_target(member_trial: Member, member_parent: Member):
    """
    Modifies the values 'member' relative to its parent.

    :param member_trial: The member to be modified.
    :param member_parent: The parent member used as a reference.
    """
    for chromosome_trial, chromosome_parent in zip(member_trial.chromosomes, member_parent.chromosomes):
        if chromosome_trial.real_value > chromosome_trial.ub:
            chromosome_trial.real_value = (chromosome_trial.ub + chromosome_parent.real_value) / 2
        elif chromosome_trial.real_value < chromosome_trial.lb:
            chromosome_trial.real_value = (chromosome_trial.lb + chromosome_parent.real_value) / 2


def boundary_resampling(trial, population, fitness_fun, base_vector_schema, optimization_type, mutation_y, mutation_f):
    """
    Tries to re-generate member 10 times, if not successful the Lamarckian repair method is used.

    :param trial: Trial population.
    :param population: Population of the original members.
    :param base_vector_schema: Base vector schema for choosing the base vector in mutation.
    :param optimization_type: Optimization type for choosing the best member.
    :param mutation_y: How many original member will be used in the mutation.
    :param mutation_f: Mutation factor.
    """
    for member_trial, member_parent in zip(trial.members, population.members):
        #10 times we try to re-generate feasible member if given member is not feasible
        for i in range(0, 10, 1):
            is_not_valid = False
            # We check if any of member's chromosome is outside of boundaries
            for chromosome_trial in member_trial.chromosomes:
                if chromosome_trial.real_value > chromosome_trial.ub or chromosome_trial.real_value < chromosome_trial.lb:
                    is_not_valid = True
            if is_not_valid:
                #Choosing of base vector based on base vector schema
                diff_members = random.sample(population.members.tolist(), 2 * mutation_y)
                if base_vector_schema == BaseVectorSchema.RAND:
                    base_vector = random.choice(population.members.tolist())
                elif base_vector_schema == BaseVectorSchema.CURRENT:
                    base_vector = member_parent
                elif base_vector_schema == BaseVectorSchema.BEST:
                    best_member = get_best_member(optimization_type, population)
                    base_vector = best_member
                else:
                    raise ValueError("Unknown base vector schema.")
                member_trial = mutation_ind(base_vector, diff_members, mutation_f)
        #We check if after 10 times re-generated member is feasible, if not we use Lamarckian repair (projection)
        is_not_valid = False
        for chromosome_trial in member_trial.chromosomes:
            if chromosome_trial.real_value > chromosome_trial.ub or chromosome_trial.real_value < chromosome_trial.lb:
                is_not_valid = True
        if is_not_valid:
            boundary_projection(member_trial, fitness_fun, 1)


def boundary_conservative(member_trial: Member, base_vector: Member):
    """
    Modifies the values 'member' relative to its base vector.

    :param member_trial: The member to be modified.
    :param base_vector: The base vector used as a reference.
    """
    for chromosome_trial, chromosome_base in zip(member_trial.chromosomes, base_vector.chromosomes):
        if chromosome_trial.real_value > chromosome_trial.ub or chromosome_trial.real_value < chromosome_trial.lb:
            chromosome_trial.real_value = chromosome_base.real_value


def boundary_projection_to_base(member_trial: Member, base_vector: Member):
    """
    Modifies the values 'member' relative to its base vector.

    :param member_trial: The member to be modified.
    :param base_vector: The base vector used as a reference.
    """
    alpha = 1.0
    eps = 1e-12
    #Array of member chromosomes
    m = np.array([ch.real_value for ch in member_trial.chromosomes], dtype=float)
    #Array of base vector chromosomes
    b = np.array([ch.real_value for ch in base_vector.chromosomes], dtype=float)
    #Arrays of boundaries
    lb = np.array([ch.lb for ch in member_trial.chromosomes], dtype=float)
    ub = np.array([ch.ub for ch in member_trial.chromosomes], dtype=float)
    for i in range(len(m)):
        d = m[i] - b[i]
        if abs(d) < eps:
            continue
        if m[i] > ub[i]:
            alpha_i = (ub[i] - b[i]) / d
        elif m[i] < lb[i]:
            alpha_i = (lb[i] - b[i]) / d
        else:
            continue
        alpha = min(alpha, alpha_i)
    for i in range(len(m)):
        m[i] = b[i] + alpha * (m[i] - b[i])


#Implemented in detpy by default
def boundary_reflection_back(member: Member):
    """
    Modifies the values of `member` in-place.

    :param member: The member to be modified.
    """
    for chromosome in member.chromosomes:
        range_i = chromosome.ub - chromosome.lb
        if chromosome.real_value > chromosome.ub:
            chromosome.real_value = chromosome.ub - (chromosome.real_value - chromosome.ub) + int(
                (chromosome.real_value - chromosome.ub) / range_i) * range_i
        elif chromosome.real_value < chromosome.lb:
            chromosome.real_value = chromosome.lb - (chromosome.lb - chromosome.real_value) + int(
                (chromosome.lb - chromosome.real_value) / range_i) * range_i
