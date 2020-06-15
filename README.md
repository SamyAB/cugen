# CuGen : CUDA Genetic Algorithm

CuGen is a Boolean satisfiability problem approximative solver, using a genetic algorithm to explore the solution space, developped using CuPy

## Formula

The formulae solved by CuGen are in the Cunnjonctif Normal Form (CNF), written in the DIMACS format

## Usage

```shell
cugen /path/to/dimacs.cnf 10000 200 0.25 0.5
```

Where:
- /path/to/dimacs.cnf is the path to the fomula written in the DIMACS format
- 10000 is the maximum number of genrations (iterations) the solver will do before stopping (It may stop before reaching this number of generations if the global solution of the formula is found)
- 200 is the number of individuals (solutions) in each generation
- 0.25 is the probability of a mutation (a change in the solution values) happening
- 0.5 is the selection ratio, meaning that only half the population of a given generation will survive to breed a new generation

