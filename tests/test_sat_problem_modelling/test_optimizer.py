from cugen.sat_problem_modelling.optimizer import generate_random_first_generation


def test_optimize_should_call_the_right_functions_in_the_right_order():
    # Given

    # When

    # Then
    pass


def test_generate_random_first_generation_returns_population_with_the_right_shape():
    # Given
    number_of_individuals = 100
    number_of_literals = 40

    # When
    initial_population = generate_random_first_generation(number_of_individuals, number_of_literals)

    # Then
    assert initial_population.shape == (number_of_individuals, number_of_literals)
