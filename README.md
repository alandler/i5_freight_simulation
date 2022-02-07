# i5_freight_simulation
Simulate freight demand and electric vehicle charging along the I-5

# Objective

This repository uses object-oriented programming to create a customizable simulation framework of vehicle travel. Users can input parameters of their scenario and run unique simulations.

The repository is set up for the I-5 use case.

# How to run a simulation

Use the interactive jupyter notebook to set simulation parameters.

Files required:
- data/distances.csv
- data/stations.csv

* test_{} is used for a simpler test case

# Testing

Run <code>py test.py</code> to run unit tests for the library. The test classes are divided according to the aspect of the simulation that they test.
- Data
- Graph
- Simulation
