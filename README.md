# i5_freight_simulation
Simulate freight demand and electric vehicle charging along the I-5

# Collaboration
You might need some configuration on your github accounts.
1. Open the terminal.
2. <code>cd</code> to navigate to the directory where you want the code to be
3. <code> git clone git@github.com:alandler/i5_freight_simulation.git </code> to clone the repo. 
    if this throws an error, follow instructions to configure git
4. <code> git checkout -b [your_branch_name] </code> to create a new branch.
5. Once you make edits run 
    1. <code> git add [filename1] [filename2] ... </code> to add changes. <code> git add -A </code> will add all changes.
    2. <code> git commit -m [message to describe changes; very short] </code>
    3. <code> git push origin [your_branch_name] </code> to push to github.

# Objective

This repository uses object-oriented programming to create a customizable simulation framework of vehicle travel. Users can input parameters of their scenario and run unique simulations.

The repository is set up for the I-5 use case.

# How to run a simulation

* Use the interactive jupyter notebook.

1. Create a simulation <code>sim = Simulation(station_G)</code> where <code>station_G = get_station_G()</code>

> Files required for station_G:
> - data/distances.csv
> - data/stations.csv

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
