Feature: Solving SAT problems with cugen

  Scenario: run the solver on a simple SAT CNF formula
    Given a simple CNF formula
    When the solver is run with set hyper parameters
    Then the solver finds the perfect solution