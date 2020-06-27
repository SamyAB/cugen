Feature: Reading a DIMACS format file

  Scenario: read DIMACS files from toughsat generator
    Given an input file generated from toughsat
    When the dimacs reader transforms the input file to a cugen formula
    Then the cugen formula is the right formula