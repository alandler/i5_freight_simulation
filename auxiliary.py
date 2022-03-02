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

