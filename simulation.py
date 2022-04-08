from multiprocessing.sharedctypes import Value
import networkx as nx
import numpy as np
import random

from tqdm import tqdm
from datetime import datetime
import pickle

# File imports
from data import select_dataset, get_station_g, ingest_electricity_data, set_random_speed_columns
from replicate_graph import layer_graph
from vehicle import Vehicle

class Simulation():
    '''Create a class for a simulation'''
    
    #### Init/Graph Mutation #### 
    def __init__(self, dataset, simulation_length = 24, battery_interval = 25, km_per_percent = 1.15):

        # electricity_data
        # TODO: currently simulation sums over full grid, rather than per state
                # Update the ingest_electricity_data (in data.py), get_electricity_metric (in simulation.py), and record_data (in simulation.py) once this is updated. - this requires a way to see what region each station is in.
                #Also include posibility to change the seasons... right now it is just always summer
        self.state_electricity_limits = {"CA": ingest_electricity_data()[1]}
        
        # data
        self.dataset = dataset
        self.stations_df, self.distances_df = select_dataset(dataset)

        # graphs
        self.station_g = get_station_g(stations_df, distances_df)
        self.battery_g = layer_graph(self.station_g, battery_interval, km_per_percent)

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

    def add_dst(self, dst, score):
        ''' add destination node and desirability score '''
        if score<0: # no negative scores
            return
        self.dst_dict[dst] = score

    def add_src(self, src, src_distr):
        ''' Add src node to list and augments the node in station_g with the src'''
        if sum(1 for number in src_distr if number < 0) > 0: # no negative scores
            return
        self.src_dict[src] = src_distr

    def random_srcs(self, flow_mean=50, flow_std=20):
        for node in self.station_g.nodes:
            hour_factors = np.array([.1, .1, .1, .1, .2, .4, .5, .8, .9, 1, .9, .8,
                            .7, .6, .8, .8, .7, .6, .5, .4, .3, .2, .1, .1])
            hour_factors += np.random.normal(0,.07,24)
            flow = np.random.normal(flow_mean,flow_std,1)
            source_distribution = flow[0]*hour_factors
            source_distribution = source_distribution.astype(int)
            self.add_src(node, source_distribution)
    
    def random_dsts(self):
        for node in self.station_g.nodes:
            score = -1
            while score < 0 or score > 10:
                score = np.random.normal(5,2,1)[0]
            self.add_dst(node, score)
            
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
    def get_random_destination(self, n, src):
        ''' Gets random destination according to probability distribution of scores'''
        dst_list = [key for key in self.dst_dict if key != src]
        desirability_score_list= [self.dst_dict[key] for key in self.dst_dict if key != src]
        total_score = sum(desirability_score_list)
        destination_probabilities = [desirability_score_list[i]/total_score for i in range(len(desirability_score_list))]
        random_destinations = np.random.choice(dst_list, size = n, replace = True, p=destination_probabilities)
        return random_destinations
    
    #################### Metrics ####################
    def calculate_metrics(self):
        self.metrics = {"station_utilization_disp_of_avg": self.get_station_utilization_disp_of_avg(),
        "station_utilization_avg_of_disp": self.get_station_utilization_avg_of_disp(),
        "electricity": self.get_electricity_metric()}

    def get_station_utilization_disp_of_avg(self):
        ''' Uses average cars in each station at each timestep to produce utilization metric
        - uses average cars in each station (over all timesteps).
        - takes difference in average utilization of the top 20% and lower 20% stations'''
        
        #get the average # of cars at each station
        cars_avg = np.average(np.array(self.data["num_cars_at_station"]), axis=0)
        
        #get the capacity of each station (sorted by station, just in case)
        physical_capacity_dict = nx.get_node_attributes(self.station_g,'physical_capacity')
        physical_capacity = [i for _,i in sorted(zip(physical_capacity_dict.keys(),physical_capacity_dict.values()))]

        #get the average utilization rate of each station
        utilization = [i / j for i, j in zip(cars_avg, physical_capacity)]
        utilization.sort()

        #Get average usage of the top 20% most used and the lower 20% least used.
        u_lower_20 = np.mean(utilization[:int(np.floor(len(utilization)*1/5))])
        u_upper_20 = np.mean(utilization[int(np.ceil(len(utilization)*4/5)):])
        
        #return dispersion of average use
        return u_upper_20 - u_lower_20
        
    def get_station_utilization_avg_of_disp(self):
        ''' Uses average cars in each station at each timestep to produce utilization metric
        - takes difference in average utilization of the top 20% and lower 20% stations (called dispersion) for each timestep 
        - takes average dispersion (over all timesteps)'''
        
        #get number of cars at each station
        cars_at_station = np.array(self.data["num_cars_at_station"])
            
        #get physical capacity of each station
        physical_capacity_dict = nx.get_node_attributes(self.station_g,'physical_capacity')
        physical_capacity = [i for _,i in sorted(zip(physical_capacity_dict.keys(),physical_capacity_dict.values()))]
        
        #get the dispersion (top20%-bottom20%) for each time step
        disp = []
        for time in range(0,len(cars_at_station)):
            this_usage = [i / j for i, j in zip(cars_at_station[time], physical_capacity)]
            this_usage.sort()
            u_lower_20 = np.mean(this_usage[:int(np.floor(len(this_usage)*1/5))])
            u_upper_20 = np.mean(this_usage[int(np.ceil(len(this_usage)*4/5)):])
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

        src_dest_df = self.distances_df[(self.distances_df['OriginID']==int(src)) & (self.distances_df['DestinationID']==int(dst))] # Access table with speeds for the src, dst pair 
        if len(src_dest_df)==0:
            print(src, dst, h_index)
        avg_speed = src_dest_df.iloc[0]["speed_"+str(h_index)] # Get speed at time-of-day = h_index (use "speed_"+h_index) 
        time = src_dest_df.iloc[0]["Total_Kilometers"]/avg_speed # Convert to time... d=rt t=d/r

        # update battery graph
        # with open("test.txt", "a") as myfile:
        src_labels = [src+"_"+str(layer)+"_"+"out" for layer in self.battery_layers]
        for src_label in src_labels:
            for dst_label in [str(dst) + "_" + str(battery_layer)+ "_in" for battery_layer in self.battery_layers]:
                try:
                    # myfile.write(src_label + "," + dst_label + "," + str(time) + "," +  str(self.station_g[src][dst]['weight']) + "\n")
                    self.battery_g[src_label][dst_label]['weight'] = time
                except:
                    continue
        
        # store in station graph just in case
        self.station_g[src][dst]['weight'] = time

        return self.battery_g, self.station_g # redundant

    def update_hourly_road_time(self, h_index):
         ''' For each _out to _in edge in the graph update edge time '''
         for edge in self.station_g.edges:
            self.change_hourly_road_time(edge[0], edge[1], h_index)

    def update_charging_times(self):
        for node in self.station_g.nodes:
            additional_wait_time = len(self.battery_g.nodes[node]["queue"]) * random.gauss(75/self.station_g.nodes[node]["charging_rate"], .5)
            self.add_charger_wait_time(node, additional_wait_time)
        
    def add_charger_wait_time(self, station, time):
        '''Mutates the graph G to add "time" to the edges between the station _in to _out'''
        for in_battery_level in self.battery_layers: # all start levels 0 to 100
            in_label = station + "_"+ str(in_battery_level) + "_in"
            for out_label in self.battery_g.neighbors(in_label):
                if "_out" in out_label and self.battery_g[in_label][out_label]["time"] != 0: # not sink and doesn't go straight through without charging
                    self.battery_g[in_label][out_label]["weight"] = self.battery_g[in_label][out_label]["time"] +  time
        return self.battery_g

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
            destinations = self.get_random_destination(num_trucks_released, src)
            for truck_i in range(num_trucks_released): # get shortest paths for each truck
                truck = Vehicle(self, src, destinations[truck_i], self.simulation_index)
                self.vehicle_list.append(truck)
                shortest_path = nx.shortest_path(self.battery_g, src, destinations[truck_i])
                truck.path = shortest_path
                truck.simulation = self

    def run(self):
        ''' Run the simulation; return the statistics '''
        for h_step in tqdm(range(self.simulation_length)): # go for simulation_length in hours
            self.update_hourly_road_time(h_step)
            for i_step in range(int(1/self.time_interval)): # go in interval segments
                self.step(h_step,i_step)
                self.simulation_index += 1
                self.record_data()
                self.update_charging_times()
            self.simulation_hour_index += 1

            # for each src dst pair 
        self.calculate_metrics()
        # self.save_simulation()
        return self.metrics

    def save_simulation(self):
        ''' Saves the simulation object as a pickle file.
        The nomenclature is the dataset plus the current datetime'''
        def save_object(obj, filename):
            with open(filename, 'wb') as outp:  # Overwrites any existing file.
                pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
        file = self.dataset + "_" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        save_object(self, file + ".pkl")

    def record_data(self):
        ''' Record vehicles at each charging node, electricity grid total usage, and queues'''
        node_car_total = {node:0 for node in list(self.station_g.nodes)}
        for vehicle in self.vehicle_list:
            if "in" in vehicle.location[0] and "out" in vehicle.location[1]: #charging edge
                charging_station_name = vehicle.location[0].split("_")[0]
                node_car_total[charging_station_name] += 1

        electricity_use = 0
        num_cars_at_station = [node_car_total[key] for key in sorted(node_car_total.keys())]
        num_vehicles_total = len([v for v in self.vehicle_list if v.travel_time == 0])

        for node in node_car_total:
            electricity_use += min(node_car_total[node],self.stations_df.set_index("OID_").loc[int(node)]["physical_capacity"])*150
        
        # Append to metrics attribute
        self.data["num_vehicles_total"].append(num_vehicles_total)
        self.data["num_cars_at_station"].append(num_cars_at_station)
        self.data["total_kw"].append(electricity_use)

if __name__ == "__main__":
    stations_df, distances_df = select_dataset("wcctci")
    simulation_length = 12
    battery_interval = 20
    km_per_percent = 3.13
    sim = Simulation("wcctci", simulation_length, battery_interval, km_per_percent)
    sim.random_srcs(55,12)
    sim.random_dsts()
    metrics = sim.run()
