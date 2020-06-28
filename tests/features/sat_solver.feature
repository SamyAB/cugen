Feature: Solving SAT problems with nugen

  Scenario: run the solver on a simple SAT CNF formula
    Given a simple CNF formula
    When the solver is run with set hyper parameters
    Then the solver finds the perfect solution

  Scenario: read a DIMACS file and solves the SAT formula
    Given a DIMACS CNF file
    When the SAT solver command is run
    Then the solver finds the perfect solution
