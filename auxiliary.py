# Useless functions that I don't feel like deleting because I am emotionally attached

def layer_graph_simple(graph, increment = 25):
    num_layers = int(100/increment + 1)
    
    # build out duplicate vertices for all layers, labelled
    output_graph = nx.DiGraph()
    nodes = [str(vertex) + "_" + str(layer*increment) for layer in range(num_layers) for vertex in list(graph)]
    output_graph.add_nodes_from(nodes)
    
    # iterate over existing roads in the input graph
    for edge in list(graph.edges):
        src = edge[0]
        dst = edge[1]
        road_weight = graph.get_edge_data(src, dst)['weight']
        road_len = graph.get_edge_data(src, dst)['length']
        
        # for each edge, create all pairwise edges between the src and dest at all appropriate battery levels
        for src_layer in range(num_layers):
            src_battery = src_layer*increment
            src_label = str(src) + "_" + str(src_layer*increment)
            if miles_per_*src_battery < road_len: # car won't make it to the next stop
                continue
            for dst_layer in reversed(range(num_layers)):  # go top to bottom. 
                dst_battery = dst_layer*increment
                dst_label = str(dst) + "_" + str(dst_layer*increment)
                if dst_battery > src_battery: # charging occurs at the given stop.
                    charging_weight = (dst_battery-src_battery)*charging[str(src)]
                    output_graph.add_edge(src_label, dst_label, weight = charging_weight + road_weight)
                elif dst_battery < src_battery: # proceed without charging.
                    output_graph.add_edge(src_label, dst_label, weight = road_weight)
                    break
                else: # edge from self not possible.
                    continue
    
    return output_graph
    
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

def get_location_breakdown(locations):
    road = 0
    queue = 0
    for location in locations:
        if "in" in location[0] and "out" in location[1]:
            queue+=1
        if "out" in location[0] and "in" in location[1]:
            road+=1
    return (road, queue)

percent_time_in_queues = []
raw_time_in_queues = []
for v in res.vehicle_list:
    if v.locations != []:
        road, queue = get_location_breakdown(v.locations)
        percent_queue = queue/(road+queue)
        raw_time_in_queues.append(queue)
    else:
        percent_queue = None
    percent_time_in_queues.append(percent_queue)
percent_time_in_queues = np.array(percent_time_in_queues)
percent_time_in_queues = percent_time_in_queues[percent_time_in_queues != np.array(None)]
percent_time_in_queues.mean()
np.array(raw_time_in_queues).mean()/5