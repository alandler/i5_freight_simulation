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
            if miles_per_percent*src_battery < road_len: # car won't make it to the next stop
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
    

