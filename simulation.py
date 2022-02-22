import networkx as nx
import numpy as np
# import pandas as pd
import random
from data import get_station_G
from replicate_graph import layer_graph
import sys
import re

class Simulation():
    '''Create a class for a simulation'''
    
    #### Init/Graph Mutation #### 
    def __init__(self, station_G, simulation_length = 24):

        # graphs
        self.station_G = station_G
        self.battery_G = layer_graph(station_G)

        # intervals
        self.time_interval = .2 # hours
        self.battery_interval = 25 # % charge
        self.num_layers = 100//self.battery_interval+1 # minimum 2
        self.battery_layers = [self.battery_interval*l for l in range(self.num_layers)] # always includes 0 and 100

        # simulation parameters: static
        self.num_batches = 10 
        self.average_demand = 10 
        self.simulation_length = simulation_length # in hours 

        # simulation data: dynamic
        self.vehicle_list = [] # all vehicles released over the cours of the simulation
        self.simulation_index = 0 # increments each self.time_interval
        self.simulation_hour_index = 0 # increments each hour
        self.src_dict = {} # veh/hr at each hour
        self.dst_dict = {} # desirabiliy score 0-10

        # metrics
        self.metrics = None
        self.state = {}

    def create_station_G(self, mph=55):
        '''Takes in the node and edge csv and creates the self.station_G'''
        self.station_G = get_station_G()

    def add_dst(self, dst, score):
        ''' add destination node and desirability score '''
        if score<0: # no negative scores
            return
        self.dst_dict[dst] = score

    def add_src(self, src, src_distr):
        ''' Add src node to list and augments the node in station_G with the src'''
        if sum(1 for number in src_distr if number < 0) > 0: # no negative scores
            return
        self.src_dict[src] = src_distr
    
    def generate_src_dst_nodes(self): # Arbitrary function for testing - will delete later
        ''' Generate src and dst nodes. Assign the hour distribution to the node in G
        Default 1 and 464, the southernmost and northenmost nodes'''
        self.add_src(1, [10 for x in range(24)])
        self.add_dst(464, 4)

    #################### Observers ####################
    def get_simulation_hour_of_day(self):
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
        if edge_label not in paths:
            return 0 
        return paths[edge_label]
    
    #################### Producers ####################
    def get_random_destination(self, n):
        ''' Gets random destination according to probability distribution of scores'''
        dst_list = list(self.dst_dict.keys())
        desirability_score_list= list(self.dst_dict.values())
        total_score = sum(desirability_score_list)
        destination_probabilities = [desirability_score_list[i]/total_score for i in range(len(desirability_score_list))]
        random_destinations = np.random.choice(dst_list, size = n, replace = True, p=destination_probabilities)
        return random_destinations

    #################### Augmentation Mutators ####################
    def update_travel_times(self):
        ''' Based on the number of cars on a given segment, update the travel time on the road according to some formula
        TODO what formula?? How do we include uncertainty? TODO DISTANCE SEGMENT ROADS
        - include rest stops
        - include temporal demand changes'''

        # iterate over all vehicles to get current locations (would concatenating an array and summming columns be easier here?)
        new_state = {}
        for vehicle in self.vehicle_list:
            location = vehicle.segmented_path[self.simulation_index]
            if location in new_state:
                new_state[location]+=1
            else:
                new_state[location] = 1

        # check congestion, capacities
        for location in new_state:
            if type(location) is tuple: # road
                time = 5 ############################################## TODO TODO TODO ##############################################
                self.add_road_time(location[0], location[1], time)
            else: # station
                physical_capacity = nx.get_node_attributes(self.station_G,'physical_capacity')[location]
                if new_state[location] > physical_capacity:
                    time = (physical_capacity-new_state[location])*2 # add two hours per truck in the line
                    self.add_charger_wait_time(location[0], location[1]) 
        pass
    
    def randomize_demand(mu, std):
        ''''''
        return
    
    # TODO: MAKE SURE SINKS ALWAYS 0
    def add_road_time(self, src, dst, time):
        '''Mutates the graph G to add time (from baseline) along edges _out to _in'''
        src_labels = [src+"_"+str(layer)+"_"+"out" for layer in self.battery_layers]
        for src_label in src_labels:
            for dst_label in self.battery_G.neighbors(src_label): # get all outgoing edges
                if "_in" in dst_label:
                    self.battery_G[src_label][dst_label]['weight'] = self.station_G[src][dst]['time'] + time
        return self.battery_G

    # TODO: MAKE SURE SINKS ALWAYS 0
    def add_charger_wait_time(self, station, time):
        '''Mutates the graph G to add "time" to the edges between the station _in to _out'''
        for in_battery_level in self.battery_layers: # all start levels 0 to 100
            in_label = station + "_"+ str(in_battery_level) + "_in"
            for out_label in self.battery_G.neighbors(in_label):
                if "_out" in out_label:
                    if self.battery_G[in_label][out_label]["weight"] != 0:
                        self.battery_G[in_label][out_label]["weight"] = self.battery_G[in_label][out_label]["time"] +  time
        return self.battery_G

    #################### Simulation Run ####################
    def step(self, h_step, i_step):
        ''' one simulation step '''
        hour = h_step%24 # get time of day (assuming hour intervals)

        # step each vehicle progress
        for vehicle in self.vehicle_list:
            vehicle.step()

        # iterate over all source nodes, release x trucks according to their hourly distribution
        for src in self.src_dict:
            num_trucks_released = int(round(self.src_dict[src][hour]*self.time_interval, 0)) # TODO: inject randomness
            destinations = self.get_random_destination(num_trucks_released)
            for truck_i in range(num_trucks_released): # get shortest paths for each truck
                truck = Vehicle(self, src, destinations[truck_i], self.simulation_index)
                self.vehicle_list.append(truck)
                shortest_path = nx.shortest_path(self.battery_G, src, destinations[truck_i])
                truck.path = shortest_path
                truck.simulation = self

    def run(self):
        ''' Run the simulation; return the statistics '''
        for h_step in range(self.simulation_length): # go for simulation_length in hours
            for i_step in range(int(1/self.time_interval)): # go in interval segments
                self.step(h_step,i_step)
                self.simulation_index += 1
                #update metrics?
            self.simulation_hour_index += 1
        
        return self.metrics
    
class Vehicle():
    '''Create a vehicle to store attributes'''
    def __init__(self, simulation, src, dst, start_time = 0):

        # initialized
        self.start_time = start_time # TODO: extend 0-24 range
        self.src = src
        self.dst = dst
        self.simulation = simulation

        # calculated
        self.path = self.get_shortest_path()
        self.segmented_path = time_segment_path(self.simulation.battery_G, self.start_time, 
            self.path, self.simulation.time_interval, self.simulation.simulation_length)
        
        # not currently updated
        self.locations = []
        self.path_i = 0
        self.location = self.path[0] # (src, dst)
        self.distance_along_segment = None # km travelled so far
    
    def step(self):
        '''Increment the location tracking'''
        # time change
        time_interval = self.simulation.time_interval

        # get current edge travel speed, km length OR charging rate
        src = self.location[0]
        dst = self.location[1]

        road_travel_time = self.simulation.battery_G[src][dst]["weight"]
        road_length = self.simulation.battery_G[src][dst]["length"]

        time_left = road_travel_time * self.distance_along_segment/road_length

        # if km remaining > d, same location
        if time_left > time_interval:
            self.locations.apppend(self.location)
        else: # else transition to next edge with some time remaining
            over_time = time_interval - time_left
            self.set_next_location()

            road_travel_time = self.simulation.battery_G[self.location[0]][self.location[1]]["weight"]
            self.distance_along_segment = over_time/road_travel_time*self.simulation.battery_G[self.location[0]][self.location[1]]["length"]

        self.locations.apppend(self.location)
    
    def set_next_location(self, i = None):
        ''' increment mile location '''
        if i != None:
            i = self.path.index(self.location)
        new_location = self.path[i+1]
        self.location = new_location
        return new_location

    def get_shortest_path(self):
        '''Calculate shortest path'''
        G = self.simulation.battery_G
        return nx.shortest_path(G, self.src, self.dst)
    
    def recalculate_path(self):
        '''Recalculate path'''
        raise NotImplementedError

def distance_segment_path(station_G, path, seg_dis):
    '''Returns a modified station_G including paths segmented into equidistant segments.
    The splits minimize the difference from seg_dis.'''
    for i in range(len(path)-1):
        src = path[i]
        dst = path[i+1]

        # get (or set) segment_length and num_segments
        try:
            segment_length = station_G[src][dst]['segment_length']
            num_segments = station_G[src][dst]['num_segments']
        except:
            total_length = station_G[src][dst]['length']
            num_segments = round(total_length/seg_dis, 0)
            segment_length = total_length/num_segments
            station_G[src][dst]['segment_length'] = segment_length
            station_G[src][dst]['num_segments'] = num_segments

        # remove src-dst connection
        station_G.remove_edge(src, dst)

        current_src = src
        current_dst = dst
        # connect src->dst via bi-directional segments
        for i in range(num_segments):
            if i == num_segments-1:
                current_dst = dst
            else:
                current_dst = src + "_" + dst +"_s" + str(i) # nomenclature: src_dst_s1, src_dst_s2
            station_G.add_edge(current_src,current_dst)
            station_G.add_edge(current_dst, current_src)
            current_src = current_dst

    return station_G

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
