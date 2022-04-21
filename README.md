# i5_freight_simulation
Simulate freight demand and electric vehicle charging along the I-5

# Cloning
You might need some configuration on your github accounts.
1. Open the terminal.
2. <code>cd</code> to navigate to the directory where you want the code to be
3. <code> git clone git@github.com:alandler/i5_freight_simulation.git </code> to clone the repo. 
    if this throws an error, follow instructions to configure git

# Objective

This repository uses object-oriented programming to create a customizable simulation framework of vehicle travel. Users can input parameters of their scenario and run unique simulations.

The repository is set up for the I-5 use case.

# How to run a simulation

* Use the interactive jupyter notebook.

1. Declare variables
    <code> simulation_length = 24
        battery_interval = 20
        km_per_percent = 3.13
        stations_path = "data/wcctci_stations-updated.csv"
        distances_path = "data/wcctci_coord_distances.csv"
     </code>
2. Create a simmulation object: 
    <code> sim = Simulation("wcctci_updated_paths", stations_path, distances_path, simulation_length, battery_interval, km_per_percent) </code>
3. Add demand modeling in one of two ways:
    1. <code>sim.add_demand_nodes()</code> will add demand with supernodes at I-5 popular locations (LA, SF, etc)
    2. Create source and destination demands
        - <code>sim.add_dst(dst, score)</code> where 0 is least desiriable and 10 is most.
        - <code>sim.add_src(src, src_distr)</code> where <code>src_distr</code> is a length-24 array with each item is a demand in trucks per hour at that          hour of day (12am, 1am, ...).
4. Call <code> metrics = sim.run()</code> to run the simulation. 
5. Load a saved simulation from a pickle using: 
            <code> 
            with open('trials/' + pickle_file, 'rb') as inp:
            res= pickle.load(inp)
            </code> 

# Testing

Run <code>py test.py</code> to run unit tests for the library. The test classes are divided according to the aspect of the simulation that they test.
- Data
- Graph
- Simulation
