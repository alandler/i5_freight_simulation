from calendar import c
import networkx as nx

class Vehicle():
    '''Create a vehicle to store attributes'''
    def __init__(self, simulation, src, dst, start_time = 0):

        # initialized
        self.start_time = start_time # misnomer: is actually start_index
        self.src = src
        self.dst = dst
        self.simulation = simulation

        # calculated
        self.path = self.get_shortest_path()
        self.baseline_battery_path_time = None
        self.baseline_time = self.get_baseline_travel_time()

        self.locations = [(self.path[0],self.path[1])]
        self.path_i = 0
        self.location = (self.path[0],self.path[1]) # (src, dst)
        self.distance_along_segment = 0 # km or battery % travelled so far
        self.travel_time = 0 # total time in transit btw origin and destination
        self.travel_delay = 0 # delay from traveling in an EV

        self.queue_time = 0

        self.finished = False
    
    def step(self):
        '''Increment the location tracking'''

        if self.finished == True: # previously reached destination
            return

        time_interval = self.simulation.time_interval # start out with entire interval to travel
        while time_interval>0:
            # If the vehicle has reached its destination store travel time and travel delay
            if self.dst+"_dst" == self.location[1]:
                self.finished = True
                end_index = self.simulation.simulation_index # end simulation index
                travel_length = end_index - self.start_time # length of travel in iteration units
                self.travel_time = self.simulation.time_interval*travel_length # calculate and store total travel time
                self.travel_delay = self.travel_time - self.baseline_time # calculate and store travel delay experienced by the vehicle
                self.locations.append(self.location)
                return

            # If entering a charging node (and not just passing through), add the car to the queue
            if "_in" in self.location[0] and "_out" in self.location[1] and self.distance_along_segment == 0:
                if (self.location[0][:-2] == self.location[1][:-3]): # passing through
                    self.set_next_location()
                    self.distance_along_segment = 0
                    continue
                sink_node_label = self.location[0].split("_")[0]
                sink_node = self.simulation.battery_g.nodes[sink_node_label]
                queue = sink_node["queue"]
                if self in queue: # if in queue, check if at front and with open spots
                    if self == queue[0] and sink_node["num_vehicles_charging"] < sink_node["physical_capacity"]:
                        del self.simulation.battery_g.nodes[sink_node_label]["queue"][0] # remove from queue
                        self.simulation.battery_g.nodes[sink_node_label]["num_vehicles_charging"] += 1 # add to charging count at node
                    else:
                        self.queue_time += 1
                        break # wait until time has passed
                else: # if not in queue, add it
                    self.simulation.battery_g.nodes[sink_node_label]["queue"].append(self)
                    continue
            
            # case that it is on either a road or charging edge and has time left
            road_travel_time = self.simulation.battery_g[self.location[0]][self.location[1]]["weight"]
            if "_in" in self.location[0] and "_out" in self.location[1]: # don't use weight for charging: queue takes care of this, use time
                 road_travel_time = self.simulation.battery_g[self.location[0]][self.location[1]]["time"]
            road_length = self.simulation.battery_g[self.location[0]][self.location[1]]["length"]

            # get time_left on current segment
            if road_length == 0: # segment of length 0 could be moving through without charging or going to a sink
                time_left = 0
            else:
                time_left = road_travel_time*(1 - self.distance_along_segment/road_length)

            # determine if staying on same segment or moving to next
            if time_left > time_interval: # if km remaining > d, same segment, increment distance along
                speed = road_length/road_travel_time #km or % / hr # TODO: access speed and charging rate directly
                self.distance_along_segment += time_interval * speed
                time_interval = 0
                break # step is over
            else: # move to next segment, update parameters
                time_interval -= time_left 
                if "_in" in self.location[0] and "_out" in self.location[1]: # if leaving a charging station, decrement number of vehicles charging at that station
                    sink_node_label = self.location[0].split("_")[0]
                    self.simulation.battery_g.nodes[sink_node_label]["num_vehicles_charging"]-=1
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
        G = self.simulation.battery_g
        return nx.shortest_path(G, self.src+"_src", self.dst+"_dst", weight="weight")

    def get_baseline_travel_time(self):
        ''' Calculate the travel time a vehicle would experience if not an electric vehicle '''
        # Use graph without charging layers
        G = self.simulation.station_demand_g
        self.baseline_battery_path_time = nx.shortest_path_length(self.simulation.battery_g, self.src+"_src", self.dst+"_dst", weight="weight")

        # Return shortest path length
        return nx.shortest_path_length(G, self.src, self.dst, weight="weight")
    
    def recalculate_path(self):
        '''Recalculate path: Not needed I think'''
        raise NotImplementedError
