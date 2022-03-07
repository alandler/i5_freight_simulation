import networkx as nx
import numpy as np
# import pandas as pd
import random
import sys
import re

# File imports
from data import get_station_G, stations_df, ingest_electricity_data, distances_df, set_random_speed_columns
from replicate_graph import layer_graph
from vehicle import Vehicle

class Simulation():
    '''Create a class for a simulation'''
    
    #### Init/Graph Mutation #### 
    def __init__(self, station_G, simulation_length = 24, battery_interval = 25, km_per_percent = 1.15):

        # electricity_data
        # TODO: currently simulation sums over full grid, rather than per state
                # Update the ingest_electricity_data (in data.py), get_electricity_metric (in simulation.py), and record_data (in simulation.py) once this is updated. - this requires a way to see what region each station is in.
                #Also include posibility to change the seasons... right now it is just always summer
        self.state_electricity_limits = {"CA": ingest_electricity_data()[1]}

        # graphs
        self.station_G = station_G
        self.battery_G = layer_graph(station_G, battery_interval, km_per_percent)

        # intervals
        self.time_interval = .2 # hours
        self.battery_interval = battery_interval # % charge
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
    
    def get_station_utilization_disp_of_avg(self):
        ''' Uses average cars in each station at each timestep to produce utilization metric
        - uses average cars in each station (over all timesteps).
        - takes difference in average utilization of the top 20% and lower 20% stations'''
        
        #get the average # of cars at each station
        cars_avg = np.average(np.array(self.data["num_cars_at_station"]), axis=0)
        
        #get the capacity of each station (sorted by station, just in case)
        physical_capacity_dict = nx.get_node_attributes(self.station_G,'physical_capacity')
        physical_capacity = [i for _,i in sorted(zip(physical_capacity_dict.keys(),physical_capacity_dict.values()))]
        
        #get the average utilization rate of each station
        utilization = [i / j for i, j in zip(cars_avg, physical_capacity)].sort()
        
        #Get average usage of the top 20% most used and the lower 20% lest used.
        u_lower_20 = np.mean(utilization[:np.floor(len(utilization)*1/5)])
        u_upper_20 = np.mean(utilization[np.ceil(len(utilization)*4/5):])
        
        #return dispersion of average use
        return u_upper_20 - u_lower_20
        
    def get_station_utilization_avg_of_disp(self):
        ''' Uses average cars in each station at each timestep to produce utilization metric
        - takes difference in average utilization of the top 20% and lower 20% stations (called dispersion) for each timestep 
        - takes average dispersion (over all timesteps)'''
        
        #get number of cars at each station
        cars_at_station = np.array(self.data["num_cars_at_station"])
            
        #get physical capacity of each station
        physical_capacity_dict = nx.get_node_attributes(self.station_G,'physical_capacity')
        physical_capacity = [i for _,i in sorted(zip(physical_capacity_dict.keys(),physical_capacity_dict.values()))]
        
        #get the dispersion (top20%-bottom20%) for each time step
        disp = []
        for time in range(0,len(cars_at_station)):
            this_usage = [i / j for i, j in zip(cars_at_station[time], physical_capacity)].sort()
            u_lower_20 = np.mean(this_usage[:np.floor(len(this_usage)*1/5)])
            u_upper_20 = np.mean(this_usage[np.ceil(len(this_usage)*4/5):])
            disp.append(u_upper_20 - u_lower_20)
            
        #return the average of those dispersions
        return np.mean(disp)
    
    def get_electricity_metric(self):
        ''' takes the electricity data and limits to create the electricity metric'''
       
        #get total demanded electricity in the area
        total_electricity = self.data["total_kw"] + self.state_electricity_limits['CA']
        #see if the electricity is above 1.2 * peak of regular demand (without the trucks)
        return np.size(np.where(np.array(total_electricity) >= max(self.state_electricity_limits['CA'])*1.2))

    #################### Augmentation Mutators ####################

    def change_hourly_road_time(self, src, dst, h_index):
        '''Mutates the graph G to update time (from baseline) along edges _out to _in'''

        src_dest_df = distances_df[(distances_df['OriginID']==int(src)) & (distances_df['DestinationID']==int(dst))] # Access table with speeds for the src, dst pair 
        if len(src_dest_df)==0:
            print(src, dst, h_index)
        avg_speed = src_dest_df.iloc[0]["speed_"+str(h_index)] # Get speed at time-of-day = h_index (use "speed_"+h_index) 
        time = src_dest_df.iloc[0]["Total_Kilometers"]/avg_speed # Convert to time

        # update battery graph
        src_labels = [src+"_"+str(layer)+"_"+"out" for layer in self.battery_layers]
        for src_label in src_labels:
            for dst_label in self.battery_G.neighbors(src_label): # get all outgoing edges
                if "_in" in dst_label: # exclude sinks
                    self.battery_G[src_label][dst_label]['weight'] = time
                    self.battery_G[src_label][dst_label]['time'] = time
        
        # store in station graph just in case
        self.station_G[src][dst]['weight'] = time

        return self.battery_G, self.station_G # redundant

    def update_hourly_road_time(self, h_index):
         ''' For each _out to _in edge in the graph update edge time '''
         for edge in self.station_G.edges:
            self.change_hourly_road_time(edge[0], edge[1], h_index)

    def update_charging_times(self):
        for node in self.station_G.nodes:
            additional_wait_time = len(self.battery_G.nodes[node]["queue"]) * random.gauss(100/self.station_G.nodes[node]["charging_rate"], .5)
            self.add_charger_wait_time(node, additional_wait_time)
        
    def add_charger_wait_time(self, station, time):
        '''Mutates the graph G to add "time" to the edges between the station _in to _out'''
        for in_battery_level in self.battery_layers: # all start levels 0 to 100
            in_label = station + "_"+ str(in_battery_level) + "_in"
            for out_label in self.battery_G.neighbors(in_label):
                if "_out" in out_label and self.battery_G[in_label][out_label]["time"] != 0: # not sink and doesn't go straight through without charging
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
                self.record_data()
                self.update_hourly_road_time(h_step)
                self.update_charging_times()
                # self.print_debug()

            self.simulation_hour_index += 1

            # for each src dst pair 
        
        return self.metrics
    
    def print_debug(self):
        print([self.battery_G.nodes[node]["queue"] for node in self.station_G.nodes])

    def record_data(self):
        ''' Record vehicles at each charging node, electricity grid total usage, and queues'''
        node_car_total = {node:0 for node in list(self.station_G.nodes)}
        for vehicle in self.vehicle_list:
            if "in" in vehicle.location[0] and "out" in vehicle.location[1]: #charging edge
                charging_station_name = vehicle.location[0].split("_")[0]
                node_car_total[charging_station_name] += 1

        electricity_use = 0
        num_cars_at_station = [node_car_total[key] for key in sorted(node_car_total.keys())]
        num_vehicles_total = sum(num_cars_at_station)

        for node in node_car_total:
            electricity_use += min(node_car_total[node],stations_df.set_index("OID_").loc[int(node)]["physical_capacity"])*150
        
        # Append to metrics attribute
        self.data["num_vehicles_total"].append(num_vehicles_total)
        self.data["num_cars_at_station"].append(num_cars_at_station)
        self.data["total_kw"].append(electricity_use)

if __name__ == "__main__":
    set_random_speed_columns()
    station_G = get_station_G()
    sim = Simulation(station_G, 24, 20, 10)
    sim.add_dst("465",5)
    src_dstr = [10,20,30,40,50,60,50,40,30,20,30,40,50,60,50,40,30,20,10,5,0,0,0,0]
    sim.add_src("1",src_dstr)
    sim.run()