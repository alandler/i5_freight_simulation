import numpy as np

def get_array(sim):
    locations = np.full((len(sim.vehicle_list),int(sim.simulation_length/sim.time_interval)), fill_value="").tolist()
    for row, vehicle in enumerate(sim.vehicle_list):
        i = vehicle.start_time
        j = vehicle.start_time + len(vehicle.locations)
        str_locations = [loc[0]+":"+loc[1] for loc in vehicle.locations]
        locations[row][i:j] = str_locations
    return np.array(locations)

def get_edge_totals(arr):
#     arr = get_array(sim)
    edges={}
    for entry in arr:
        loc = entry.split(":")
        src = loc[0].split("_")[0]
        try:
            dst = loc[1].split("_")[0]
        except:
            dst = src
        if (src, dst) in edges:
            edges[(src, dst)]+=1
        else:
            edges[(src, dst)]=0
    return edges

def get_index_edge_totals(sim):
    locations = get_array(sim)
    return np.apply_along_axis(get_edge_totals, 1, locations.T)

def get_node_totals(sim):
    sim_edge_totals = get_index_edge_totals(sim)
    sim_length_indices = int(sim.simulation_length/sim.time_interval)
    node_totals = {}
    for sim_index in range(sim_length_indices):
        node_totals[sim_index] = {node: sim_edge_totals[sim_index][(node, node)] if (node, node) in sim_edge_totals[sim_index] else 0 for node in sim.station_g.nodes}
    return node_totals

def get_road_totals(sim):
    sim_edge_totals = get_index_edge_totals(sim)
    sim_length_indices = int(sim.simulation_length/sim.time_interval)
    road_totals = {}
    for sim_index in range(sim_length_indices):
        road_totals[sim_index] = {edge: sim_edge_totals[sim_index][edge] if edge in sim_edge_totals[sim_index] else 0 for edge in sim.station_g.edges}
    return road_totals

def get_node_queue_lengths(sim):
    charging_and_queue_totals = get_node_totals(sim)
    queue_totals = {}
    sim_length_indices = int(sim.simulation_length/sim.time_interval)
    for sim_index in range(sim_length_indices):
        for node in charging_and_queue_totals:
            total_minus_capacity = charging_and_queue_totals[sim_index][node] - sim.station_g.nodes[node]['physical_capacity']
            queue_totals[sim_index][node] = total_minus_capacity if total_minus_capacity>0 else 0
    return queue_totals

def get_vehicles_charging(sim):
    charging_and_queue_totals = get_node_totals(sim)
    charging_totals = {}
    sim_length_indices = int(sim.simulation_length/sim.time_interval)
    for sim_index in range(sim_length_indices):
        for node in charging_and_queue_totals:
            total_minus_capacity = charging_and_queue_totals[sim_index][node] - sim.station_g.nodes[node]['physical_capacity']
            charging_totals[sim_index][node] = -total_minus_capacity if total_minus_capacity<0 else sim.station_g.nodes[node]['physical_capacity']
    return charging_totals

def get_total_vehicles_in_queues(sim):
    queue_lengths = get_node_queue_lengths(sim)
    vehicles_in_queues = {}
    for sim_index in sim:
        vehicles_in_queues[sim_index] = sum(queue_lengths[sim_index])
    return vehicles_in_queues

def get_utilization(sim):
    charging_and_queue_totals = get_node_totals(sim)
    utilization_totals = {}
    sim_length_indices = int(sim.simulation_length/sim.time_interval+1)
    for sim_index in range(sim_length_indices):
        for node in charging_and_queue_totals:
            if sim_index in utilization_totals:
                utilization_totals[sim_index]= {}
            else:
                utilization_totals[sim_index][node] = charging_and_queue_totals[sim_index][node]/sim.station_g.nodes[node]['physical_capacity']
    return utilization_totals

