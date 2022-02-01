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
        self.time_interval = 10 # minutes
        self.battery_interval = 25 # % charge
        self.num_layers = 100//self.battery_interval+1
        self.battery_layers = [self.battery_interval*l for l in range(self.num_layers)]

        # simulation parameters: static
        self.num_batches = 10 
        self.average_demand = 10
        self.simulation_length = simulation_length

        # simulation data: dynamic
        self.vehicle_list = None
        self.simulation_index = 0
        self.src_dict = {}
        self.dst_dict = {}

    def get_simulation_hour(self):
        '''Returns the hour [0-24] of the time of day at the current simulation index'''
        return round((self.simulation_index * self.time_interval)/60, 0) % 24
    
    def get_number_vehicles(self, edge_label):
        '''Get vehicles along a given edge (all battery levels)'''
        paths = {}
        i = self.get_simulation_index()
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

    def step(self, h_step):
        ''' one simulation step '''
        hour = h_step%24 # get time of day (assuming hour intervals)

        # iterate over all source nodes, release x trucks according to their hourly distribution
        for src in self.src_dict:
            demand = self.src_dict[src][hour]
            dst_list = self.dst_dict.keys()
            desirability_score_list = self.dst_dict.values()
            destination_probabilities = list(desirability_score_list/desirability_score_list.sum())
            np.random.choice(dst_list, p=destination_probabilities)

        return 

    def run(self):
        ''' Run the simulation; return the statistics '''
        for hour in range(self.simulation_length): # go for simulation_length in hours
            for interval in range(60/self.time_interval): # go in interval segments
                self.step(hour)
                self.simulation_index += 1
        
        return self.statistics

    def add_wait_time(self, G, station, time, battery_interval):
        '''Mutates the graph G to add "time" to the edges between the station _in to _out'''
        battery_layers = [battery_interval*l for l in range(self.num_layers)]
        for in_battery_level in battery_layers: # all start levels 0 to 100
            in_label = station + "_"+ in_battery_level + "_in"
            for out_battery_level in range(in_battery_level, 100+battery_interval,battery_interval): # all ends > start
                out_label = station + "_"+ out_battery_level + "_out"
                G[in_label][out_label] += time
        return G

    def add_dst(self, dst, score):
        ''' add destination node and desirability score '''
        self.dst_dict[dst]["dst_score"] = score

    def add_src(self, src, src_distr):
        ''' Add src node to list and augments the node in station_G with the src'''
        self.src_dict[src] = src_distr

    def generate_src_dst_nodes(self):
        ''' Generate src and dst nodes. Assign the hour distribution to the node in G
        Default 1 and 464, the southernmost and northenmost nodes'''
        self.add_src(self, 1, [1 for x in range(24)])
        self.add_dst(self, 464, 4)
    
class Vehicle():
    '''Create a vehicle to store attributes'''
    def __init__(self, simulation, src, dst, start_time =0):

        # initialized
        self.start_time = start_time # TODO: extend 0-24 range
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
    
    def recalculate_path(self):
        '''Recalculate path'''
        raise NotImplementedError

def distance_segment_path(road_G, path, seg_dis):
    '''Returns a modified road_G including paths segmented into equidistant segments.
    The splits minimize the difference from seg_dis.'''
    for i in range(len(path)-1):
        src = path[i]
        dst = path[i+1]

        # get (or set) segment_length and num_segments
        try:
            segment_length = road_G[src][dst]['segment_length']
            num_segments = road_G[src][dst]['num_segments']
        except:
            total_length = road_G[src][dst]['length']
            num_segments = round(total_length/seg_dis, 0)
            segment_length = total_length/num_segments
            road_G[src][dst]['segment_length'] = segment_length
            road_G[src][dst]['num_segments'] = num_segments

        # remove src-dst connection
        road_G.remove_edge(src, dst)

        current_src = src
        current_dst = dst
        # connect src->dst via segments
        for i in range(num_segments):
            if i == num_segments-1:
                current_dst = dst
            else:
                current_dst = src + "_s" + str(i)
            road_G.add_edge(current_src,current_dst)
            current_src = current_dst

    return road_G

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