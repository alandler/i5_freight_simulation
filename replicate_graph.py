import networkx as nx
from tqdm import tqdm


def find_nearest_increment(n, increment=25):
    '''Finds the rounded down defined layer to the battery input (n), inefficiently'''
    curr = 0
    # runs through levels until hits n (or lower)
    while curr <= n-increment:
        curr += increment
    return curr

def layer_graph(graph, increment= 25, km_per_percent = 4):
    '''Creates a duplicated graph, where each node contains all battery levels both in and out. 
    Directed edges exist from out to in and in to out, the former being roads, and the latter being charging.
    The size of the graph is 3V + VE^2 '''
    num_layers = int(100/increment + 1)
    battery_layers = [increment*l for l in range(num_layers)]

    # build out in and out vertices for all layers, labelled [node_percent_in/out]
    output_graph = nx.DiGraph()
    in_nodes = [str(vertex) + "_" + str(layer*increment) +
                "_in" for layer in range(num_layers) for vertex in list(graph)]
    out_nodes = [str(vertex) + "_" + str(layer*increment) +
                 "_out" for layer in range(num_layers) for vertex in list(graph)]
    nodes = in_nodes + out_nodes
    output_graph.add_nodes_from(nodes)
    # add sinks from original
    output_graph.add_nodes_from([str(node) for node in graph.nodes()])

    # Roads: iterate over existing roads in the input graph (these form links from _out to _in)
    for edge in list(graph.edges):
        src = edge[0]
        dst = edge[1]
        road_weight = graph.get_edge_data(src, dst)['weight']
        road_len = graph.get_edge_data(src, dst)['length']

        # create all pairwise edges between the src and dest at all appropriate battery levels
        for src_layer in range(num_layers):
            src_battery = src_layer*increment  # number
            src_label = str(src) + "_" + str(src_layer* \
                            increment) + "_out"  # source node is _out

            # check battery sufficient to travel from _out to _in
            if km_per_percent*src_battery > road_len:
                # add link to next charging station at current-cost charge
                battery_cost = road_len/km_per_percent  # TODO: confirm correctness
                battery_layer = find_nearest_increment(
                    src_battery-battery_cost, increment)
                dst_label = str(dst) + "_" + str(battery_layer) + \
                                "_in"  # dst node is _in
                output_graph.add_edge(src_label, dst_label, weight=road_weight)  # _out to _in

    # Charging: iterate over nodes and connect _in to _out for all positive battery levels, and _out to sinks
    charging_rates = graph.nodes(data="charging_rate")
    for node_data in charging_rates:
        node = node_data[0]
        charging_rate = node_data[1]
        for i, src_battery_layer in enumerate(battery_layers):
            src_label = str(node) + "_" + \
                str(src_battery_layer) + "_in"  # src node is _in
            # connect to upper _out only if node is a charging node
            if charging_rate != 0:
                for dst_battery_layer in battery_layers[i:]:
                    dst_label = str(node) + "_" + \
                        str(dst_battery_layer) + "_out"
                    charging_time = (dst_battery_layer -
                                     src_battery_layer)/charging_rate
                    output_graph.add_edge(src_label, dst_label, weight=charging_time, time= charging_time) # _in to _out
                    output_graph.add_edge(dst_label, str(node), weight=0)  # _out to sink
            else:
                dst_label = str(node) + "_" + str(src_battery_layer) + "_out"
                output_graph.add_edge(src_label, dst_label, weight=0, time= charging_time)
                output_graph.add_edge(dst_label, str(node), weight=0)  # _out to sink

            output_graph.add_edge(str(node), src_label, weight=0)  # sink to _in

    return output_graph
