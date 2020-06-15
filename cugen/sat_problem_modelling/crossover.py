import cupy


def uniform_binary_crossover(first_parent_individual: cupy.ndarray,
                             second_parent_individual: cupy.ndarray) -> cupy.ndarray:
    """
    Builds a new child by randomly choosing parts of the two input parents, mimicking biological reproduction

    :param first_parent_individual: The array representing the first parent
    :param second_parent_individual: The array representing the second parent
    :return: A new individual from the mixing of the two input individuals
    """
    return cupy.where(cupy.random.uniform() <= 0.5, first_parent_individual, second_parent_individual)


def population_binary_crossover(population: cupy.ndarray, next_generation_size: int) -> cupy.ndarray:
    """
    Creates a new generation of individuals, where each individual is a child of two individuals from the input
    population. The parents are chosen randomly, and the crossover is uniform, and the new generation has a fixed size

    :param population: The parents
    :param next_generation_size: The number of individuals to generate in the new population
    :return: The new population with next_generation_size individuals
    """
    number_of_parents = population.shape[0]
    last_mating_individual_index = number_of_parents if number_of_parents % 2 == 0 else number_of_parents - 1
    minimum_number_of_matings = next_generation_size // (number_of_parents // 2) + 1

    all_mating_seasons_children = []
    for _ in range(minimum_number_of_matings):
        shuffled_population = cupy.random.permutation(population)[:last_mating_individual_index]
        pair_parents = shuffled_population[::2]
        odd_parents = shuffled_population[1::2]

        mating_season_children = cupy.array([
            uniform_binary_crossover(first_parent, second_parent)
            for first_parent, second_parent in zip(pair_parents, odd_parents)
        ])

        all_mating_seasons_children.append(mating_season_children)

    new_generation = cupy.concatenate(all_mating_seasons_children)[:next_generation_size]

    return new_generation

