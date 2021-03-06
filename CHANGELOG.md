# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.2] - 2020-06-30
### Fixed
- Ensure the DIMACS file reader crashes when it reads a literal number higher than the number of literals

## [1.0.1] - 2020-06-27
### Fixed
- Reader of DIMACS CNF formula format to handle toughsat generated formulae

## [1.0.0] - 2020-06-15
### Added
- Reader of DIMACS CNF formula format
- Boolean satisfiability problem solver based on a genetic algorithm 
- CLI to run the solver
