# i5_freight_simulation
Simulate freight demand and electric vehicle charging along the I-5

# Objective

This repository uses object-oriented programming to create a customizable simulation framework of vehicle travel. Users can input parameters of their scenario and run unique simulations.

The repository is set up for the I-5 use case.

# How to run a simulation

* Use the interactive jupyter notebook.

1. Create a simulation <code>sim = Simulation(station_G)</code> where <code>station_G = get_station_G()</code>

> Files required for station_G:
- data/distances.csv
- data/stations.csv

2. Adjust parameters
- <code> sim.simulation_length </code> defaults to 24 hours.
- <code> sim.time_interval </code> defaults to .2 hours.
- <code> sim.battery_interval </code> defaults to 25%.

3. Create source and destination demands
- <code>sim.add_dst(dst, score)</code> where 0 is least desiriable and 10 is most.
- <code>sim.add_src(src, src_distr)</code> where <code>src_distr</code> is a length-24 array with each item is a demand in trucks per hour at that hour of day (12am, 1am, ...).

4. Call <code>res = sim.run()</code> to run the simulation. <code> res </code> will be the returned metrics.

# Testing

Run <code>py test.py</code> to run unit tests for the library. The test classes are divided according to the aspect of the simulation that they test.
- Data
- Graph
- Simulation
