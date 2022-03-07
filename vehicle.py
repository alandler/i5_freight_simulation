from calendar import c
import networkx as nx

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
        # self.segmented_path = time_segment_path(self.simulation.battery_G, self.start_time, 
        #     self.path, self.simulation.time_interval, self.simulation.simulation_length)
        self.baseline_time = self.get_baseline_travel_time()

        # not currently updated
        self.locations = []
        self.path_i = 0
        self.location = (self.path[0],self.path[1]) # (src, dst)
        self.distance_along_segment = 0 # km or battery % travelled so far
        self.travel_time = 0 # total time in transit btw origin and destination
        self.travel_delay = 0 # delay from traveling in an EV
    
    def step(self):
        '''Increment the location tracking'''

        time_interval = self.simulation.time_interval # start out with entire interval to travel
        while time_interval>0:
            # If the vehicle has reached its destination store travel time and travel delay
            if self.dst == self.location[1]:
                end_index = self.simulation.simulation_index # end simulation index
                travel_length = end_index - self.start_time # length of travel in iteration units
                self.travel_time = self.simulation.time_interval*travel_length # calculate and store total travel time
                self.travel_delay = self.travel_time - self.baseline_time # calculate and store travel delay experienced by the vehicle
                return

            # If entering a charging node (and not just passing through), add the car to the queue
            if "_in" in self.location[0] and "_out" in self.location[1] and self.distance_along_segment == 0:
                if (self.location[0][:-2] == self.location[1][:-3]): # passing through
                    self.set_next_location()
                    self.distance_along_segment = 0
                    continue
                sink_node_label = self.location[0].split("_")[0]
                sink_node = self.simulation.battery_G.nodes[sink_node_label]
                queue = sink_node["queue"]
                if self in queue: # if in queue, check if at front and with open spots
                    if self == queue[0] and sink_node["num_vehicles_charging"] < sink_node["physical_capacity"]:
                        del self.simulation.battery_G.nodes[sink_node_label]["queue"][0]
                        self.simulation.battery_G.nodes[sink_node_label]["num_vehicles_charging"] += 1
                    else:
                        break # wait until time has passed
                else: # if not in queue, add it
                    self.simulation.battery_G.nodes[sink_node_label]["queue"].append(self)
                    continue
            
            road_travel_time = self.simulation.battery_G[self.location[0]][self.location[1]]["weight"]
            if "_in" in self.location[0] and "_out" in self.location[1]: # don't use weight for charging: queue takes care of this, use time
                 road_travel_time = self.simulation.battery_G[self.location[0]][self.location[1]]["time"]
            road_length = self.simulation.battery_G[self.location[0]][self.location[1]]["length"]

            # get time_left on current segment
            if road_length == 0: # segment of length 0 could be moving through without charging or going to a sink
                time_left = 0
            else:
                time_left = road_travel_time - (road_travel_time * self.distance_along_segment/road_length)

            # determine if staying on same segment or moving to next
            if time_left > time_interval: # if km remaining > d, same segment, increment distance along
                self.locations.append(self.location)
                speed = road_length/road_travel_time #km or % / hr
                self.distance_along_segment += time_interval * speed
                time_interval = 0
            else: # move to next segment, update parameters
                time_interval -= time_left 
                if "_out" in self.location[1]: # if leaving a charging station, decrement number of vehicles charging at that station
                    sink_node_label = self.location[0].split("_")[0]
                    self.simulation.battery_G.nodes[sink_node_label]["num_vehicles_charging"]-=1
                self.set_next_location()
                self.distance_along_segment = 0

        self.locations.append(self.location) # only set the location at the end of the simulation_index (even if multiple are traversed)
    
    def set_next_location(self):
        ''' increment mile location '''
        new_location = (self.path[self.path_i+1], self.path[self.path_i+2])
        self.path_i += 1
        self.location = new_location
        return new_location

    def get_shortest_path(self):
        '''Calculate shortest path'''
        G = self.simulation.battery_G
        return nx.shortest_path(G, self.src, self.dst)

    def get_baseline_travel_time(self):
        ''' Calculate the travel time a vehicle would experience if not an electric vehicle '''
        # Use graph without charging layers
        G = self.simulation.station_G

        # Return shortest path length
        return nx.shortest_path_length(G, self.src, self.dst, weight="weight")
    
    def recalculate_path(self):
        '''Recalculate path: Not needed I think'''
        raise NotImplementedError
