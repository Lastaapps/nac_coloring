# Flexible realizations existence: NP-completeness on sparse graphs and algorithms

*Petr Laštovička and Jan Legerský*

![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17594025.svg)

In this repository, we provide our code for the NAC-coloring search algorithm
described in the [paper](https://www.arxiv.org/abs/2412.13721)
*Flexible realizations existence: NP-completeness on sparse graphs and algorithms*.
We prepared a notebook `NAC_playground.ipynb` where
you can experiment with the algorithm and
`NAC_presentation.ipynb` in which you can see how we run and analyze benchmarks.
More details about the implementation can be found in the corresponding chapter
in Petr Laštovička's [bachelor thesis](http://hdl.handle.net/10467/123519).

## Setup

Python 3.12 or higher is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

You can run the tests by executing `pytest`.
The packages also contains base for Cartesian NAC-coloring search,
the related tests are skipped for now as it is not yet fully implemented
for every approach.

## Structure

- `nac` - the implementation of our NAC-coloring related algorithms and heuristics
- `stablecut` - the implementation of an algorithm to search for stable cuts
- `benchmarks` - utility files related to benchmarks - graphs loading, generation, analysis notebook utility functions
- `benchmarks/precomputed` - Results of the benchmarks as run on our hardware
- `graphs_store` - stores generated graphs of selected classes

## Graphs dataset

We list sources of the graph classes and generation tools used
to create graph datasets of specific graph classes.
Benchmarks later use these graphs to perform algorithms comparison.
All the graphs are stored in the `graphs_store` directory.
For proper description, see the thesis.

### Minimally rigid graphs

We listed all minimally rigid graphs
using [Nauty](https://pallini.di.uniroma1.it/)
with [nauty-laman-plugin](https://github.com/martinkjlarsson/nauty-laman-plugin).

We swiftly describe the folder contents:
`./nauty/minimally_rigid_all` holds all the minimally rigid graphs with the given number of vertices (up to 11 vertices).

For benchmarks, larger minimally rigid graphs were generated randomly as Nauty generates graphs
that are quite similar.
Random minimally rigid graphs are stored in the `./random/minimally_rigid` folder.

### Globally rigid graphs

Globally rigid graphs were generated randomly
using the threshold function from https://doi.org/10.1002/jgt.20196.

### NAC-critical graphs

These were generated randomly using the threshold function for NAC-coloring existence from
[paper](https://arxiv.org/abs/2510.05838) *Sharp thresholds for NAC-colourings and stable cuts in random graphs*.
For each vertex number, only graphs with at least |V(G)|/4 triangle components were kept.

## License

The code is licensed under MIT license.
