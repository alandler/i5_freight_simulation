import networkx as nx
import numpy as np
# import pandas as pd
import random
import sys
import re

# File imports
from data import get_station_G, stations_df, ingest_electricity_data, distances_df
from replicate_graph import layer_graph
from vehicle import Vehicle

class Simulation():
    '''Create a class for a simulation'''
    
    #### Init/Graph Mutation #### 
    def __init__(self, station_G, simulation_length = 24, battery_interval = 25, km_per_percent = 1.15):

        # electricity_data
        # TODO: currently simulation sums over full grid, rather than per state
        state_electricity_limits = {"CA": ingest_electricity_data()[1]}

        # graphs
        self.station_G = station_G
        self.battery_G = layer_graph(station_G, battery_interval, km_per_percent)

        # intervals
        self.time_interval = .2 # hours
        self.battery_interval = 25 # % charge
        self.num_layers = 100//self.battery_interval+1 # minimum 2
        self.battery_layers = [self.battery_interval*l for l in range(self.num_layers)] # always includes 0 and 100

        # simulation parameters: static
        self.simulation_length = simulation_length # in hours 

        # simulation data: dynamic
        self.vehicle_list = [] # all vehicles released over the cours of the simulation
        self.simulation_index = 0 # increments each self.time_interval
        self.simulation_hour_index = 0 # increments each hour
        self.src_dict = {} # veh/hr at each hour
        self.dst_dict = {} # desirabiliy score 0-10

        # metrics
        self.data = {"num_cars_at_station": [], 
                        "total_kw":[],
                        "num_vehicles_total":[]}
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

    # def add_road_time(self, src, dst, time):
    #     '''Mutates the graph G to add time (from baseline) along edges _out to _in'''
    #     src_labels = [src+"_"+str(layer)+"_"+"out" for layer in self.battery_layers]
    #     for src_label in src_labels:
    #         for dst_label in self.battery_G.neighbors(src_label): # get all outgoing edges
    #             if "_in" in dst_label:
    #                 self.battery_G[src_label][dst_label]['weight'] = self.station_G[src][dst]['time'] + time
    #     return self.battery_G

    def change_hourly_road_time(self, src, dst, h_index):
        '''Mutates the graph G to update time (from baseline) along edges _out to _in'''
        # Access table with speeds for the src, dst pair 
        src_dest_df = distances_df[(distances_df['OriginID']==src) & (distances_df['DestinationID']==dst)]
        # Get speed at time-of-day = h_index (use "speed_"+h_index) 
        avg_speed = src_dest_df["speed_"+str(h_index)][0]
        # Convert to time
        time = avg_speed*src_dest_df["Total_Kilometers"][0]

        src_labels = [src+"_"+str(layer)+"_"+"out" for layer in self.battery_layers]
        for src_label in src_labels:
            for dst_label in self.battery_G.neighbors(src_label): # get all outgoing edges
                if "_in" in dst_label:
                    self.battery_G[src_label][dst_label]['weight'] = time
                    self.battery_G[src_label][dst_label]['time'] = time
                    self.station_G[src_label][dst_label]['weight'] = time
        return self.battery_G, self.station_G

    def update_hourly_road_time(self, h_index):
        ''' For each _out to _in edge in the graph update edge time '''
        pass

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
                self.record_data() ##### TODO: build out metrics

            self.simulation_hour_index += 1

            # for each src dst pair 
        
        return self.metrics

    def record_data(self):
        ''' Record vehicles at each charging node.
        TODO Determine if we want to record vehicles at each road '''
        node_car_total = {node:0 for node in list(self.station_G.nodes)}
        for vehicle in self.vehicle_list:
            if "in" in vehicle.location[0] and "out" in vehicle.location[1]: #charging edge
                charging_station_name = vehicle.location[0].split("_")[0]
                node_car_total[charging_station_name] += 1

        electricity_use = 0
        num_cars_at_station = [node_car_total[key] for key in sorted(node_car_total.keys())]
        num_vehicles_total = sum(num_cars_at_station)

        for node in node_car_total:
            excess = node_car_total[node] - stations_df.set_index("OID_").loc[int(node)]["physical_capacity"]
            electricity_use += (node_car_total[node]-excess)*150
        
        # Append to metrics attribute
        self.data["num_vehicles_total"].append(num_vehicles_total)
        self.data["num_cars_at_station"].append(num_cars_at_station)
        self.data["total_kw"].append(electricity_use)

if __name__ == "__main__":
    station_G = get_station_G()
    sim = Simulation(station_G, 24, 20, 6)
    sim.add_dst("465",5)
    src_dstr = [10,20,30,40,50,60,50,40,30,20,30,40,50,60,50,40,30,20,10,5,0,0,0,0]
    sim.add_src("1",src_dstr)
    sim.run()