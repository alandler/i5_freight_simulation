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
    

    # TODO: handle if on charging edge INCLUDING if over physical capacity + update simulation as needed
    # TODO: handle moving over multiple edges (while loop)
    def step(self):
        '''Increment the location tracking'''

        time_interval = self.simulation.time_interval # start out with entire interval to travel
        time_left = -1 # guaranteed always to enter the while loop
        while time_left < time_interval:
            # If the vehicle has reached its destination store travel time and travel delay
            if self.dst == self.location[1]:
                end_index = self.simulation.simulation_index # end simulation index
                travel_length = end_index - self.start_time # length of travel in iteration units
                self.travel_time = self.simulation.time_interval*travel_length # calculate and store total travel time
                self.travel_delay = self.travel_time - self.baseline_time # calculate and store travel delay experienced by the vehicle
                return

            road_travel_time = self.simulation.battery_G[self.location[0]][self.location[1]]["weight"]
            road_length = self.simulation.battery_G[self.location[0]][self.location[1]]["length"]

            # get time_left on current segment
            if road_length == 0: # segment of length 0 could be moving through without charging or going to a sink
                time_left = 0
            else:
                time_left = road_travel_time * self.distance_along_segment/road_length

            # determine if staying on same segment or moving to next
            if time_left > time_interval: # if km remaining > d, same segment, increment distance along
                self.locations.apppend(self.location)
                self.distance_along_segment += time_interval/road_travel_time*self.simulation.battery_G[self.location[0]][self.location[1]]["length"]
            else: # move to next segment, update parameters
                time_interval -= time_left 
                self.set_next_location()
                time_left = self.simulation.battery_G[self.location[0]][self.location[1]]["weight"] # time_left is now the length of the next segment

        self.locations.append(self.location)
    
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
