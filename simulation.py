import networkx as nx
import numpy as np
# import pandas as pd
import random
from data import get_station_G
from replicate_graph import layer_graph

class Simulation():
    '''Create a class for a simulation'''
    
    def __init__(self, road_G, simulation_length = 24):

        # graphs
        self.road_G = road_G
        self.battery_G = layer_graph(road_G)

        # intervals
        self.time_interval = 10 
        self.battery_interval = 25
        self.num_layers = 100//self.battery_interval+1
        self.battery_layers = [self.battery_interval*l for l in range(self.num_layers)]

        # simulation parameters: static
        self.num_batches = 10 
        self.average_demand = 10
        self.simulation_length = simulation_length

        # simulation data: dynamic
        self.vehicle_list = None
        self.simulation_time = 0
    
    def get_simulation_interval(self):
        '''Returns the index of the current simulation interval'''
        return self.simulation_time//self.time_interval
    
    def get_number_vehicles(self, edge_label):
        '''Get vehicles along a given edge (all battery levels)'''
        paths = {}
        i = self.get_simulation_interval()
        for vehicle in self.vehicle_list: # iterate over all vehicles in the simulation
            vehicle_location = vehicle.segmented_path[i] 
            if vehicle_location in paths:
                paths[vehicle_location]+=1
            else:
                paths[vehicle_location]=1
        return paths[edge_label]

    def compute_travel_times():
        '''TODO: Will be used to encode uncertainty into the edge weights
        - include rest stops
        - include temporal demand changes'''
        return
    
    def create_road_G(self, mph=55):
        '''Takes in the node and edge csv and creates the self.road_G'''
        self.road_G = get_station_G()
    
    def randomize_demand(mu, std):
        ''''''
        return
    
    def add_road_time(self, G, src, dst, time):
        '''Mutates the graph G to add time along edges _out to _in'''
        out_labels = [src+"_"+layer+"_"+"out" for layer in self.battery_layers]
        for label in out_labels:
            for edge in G.out_edges(label): # get all outgoing edges
                if dst in edge[1]: 
                    G[edge[0]][edge[1]]['weight'] += time
        return G

    def run():
        return

    def add_wait_time(self, G, station, time, battery_interval):
        '''Mutates the graph G to add "time" to the edges between the station _in to _out'''
        battery_layers = [battery_interval*l for l in range(self.num_layers)]
        for in_battery_level in battery_layers: # all start levels 0 to 100
            in_label = station + "_"+ in_battery_level + "_in"
            for out_battery_level in range(in_battery_level, 100+battery_interval,battery_interval): # all ends > start
                out_label = station + "_"+ out_battery_level + "_out"
                G[in_label][out_label] += time
        return G
    
class Vehicle():
    '''Create a vehicle to store attributes'''
    def __init__(self, simulation, src, dst, start_time =0):

        # initialized
        self.start_time = start_time #between 0 and 24
        self.src = src
        self.dst = dst
        self.simulation = simulation

        # calculated
        self.path = self.get_shortest_path()
        self.segmented_path = time_segment_path(self.simulation.battery_G, self.start_time, 
            self.path, self.simlation.time_interval, self.simulation.simulation_length)
        
        # not currently updated
        self.location = None
    
    def get_shortest_path(self):
        '''Calculate shortest path'''
        G = self.simulation.battery
        return nx.shortest_path(G, self.src, self.dst)
         

def time_segment_path(G, start_time, path, time_interval, end_simulation_time):
    '''Given a vehicle, list its positions at time_interval markings'''
    
    # initialize 
    simulation_time = 0
    total_path_time = start_time
    segment_locations = []
    
    # record None for simulation time during which vehicle hasn't left
    while start_time>simulation_time:
        segment_locations.append(None)
        simulation_time+=time_interval
    
    # iterate over whole path
    for i in range(len(path)-1):
        #intialize variables
        src = path[i]
        dst = path[i+1]
        edge_time = G.get_edge_data(src, dst)['weight']
        
        # has not started traversing current edge
        curr_edge_time = 0
        
        # if the next simulation interval falls on the edge, record it
        while simulation_time <= total_path_time + edge_time:
            
            # charging: src_in to dst_out
            if src[-3:]=="_in" and src[:-3]!=dst[:-4]:
                charging_node = src.split("_")[0]
                segment_locations.append(charging_node)
            # road: src_out to dst_in
            else:
                src_node = src.split("_")[0]
                dst_node = dst.split("_")[0]
                segment_locations.append((src_node, dst_node))
            
            # increment the simulation interval 
            simulation_time += time_interval
            
        # update total path time after edge is traversed (and recorded, if needed)
        total_path_time += edge_time
    
    # add None's once at destination
    while simulation_time <= end_simulation_time:
        segment_locations.append(None)
        simulation_time+=time_interval
    
    # return the locations and the time to traverse the path (not including empty simulation time)
    return (segment_locations, total_path_time-start_time)

def single_route_simulation(G, source, destination, num_batches, average_demand):
    '''Runs a simulation of demand caused by vehicles trying to travel from source to destination.
    There will be num_batches of groups leaving at the same time. The size of the group is generated
    by a poisson distribution with average_demand'''
    all_time_segments = []
    for batch_index, batch in enumerate(range(num_batches)): # a batch is released every 15 minutes
        demand = np.random(average_demand) # random amount of trucks released at the same time in the batch
        for vehicle_index, vehicle in enumerate(range(demand)):
            starting_battery = random.random.choice([50, 100]) # TODO: parameterize the battery level choices
            starting_label = source + "_" + starting_battery + "_in" # TODO: do we always start at a charging station?
            nx.shortest_path(G,starting_label,destination)
            leading_zero_time_segmentation = [0 for x in range(batch_index)]
            time_segmentation = leading_zero_time_segmentation + time_segment_path(G,path)
            # TODO: path segmentation?
            all_time_segments.append(time_segmentation)
        # TODO post-batch actions
        # assess physical capacities (add weights to charging times)
        # assess congestion
        # assess electrical capacities (add weights to charinging times)
    return all_time_segments