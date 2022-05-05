import numpy as np
from vehicle import get_edge_type_frequencies
import pandas as pd

##################### vehicle_list_data #####################
def get_loc_array(sim, loc_type = False):
    locations = np.full((len(sim.vehicle_list),int(sim.simulation_length/sim.time_interval)), fill_value="").tolist()
    for row, vehicle in enumerate(sim.vehicle_list):
        i = vehicle.start_time
        j = vehicle.start_time + len(vehicle.locations)
        if loc_type == False:
            str_locations = [loc[0]+":"+loc[1] for loc in vehicle.locations]
        else:
            str_locations = [loc_type for loc_type in vehicle.location_types]
        locations[row][i:j] = str_locations
    return np.array(locations)

def get_edge_totals(arr):
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
    locations = get_loc_array(sim)
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
    queue_totals = {sim_index:{} for sim_index in charging_and_queue_totals}
    sim_length_indices = int(sim.simulation_length/sim.time_interval)
    for sim_index in range(sim_length_indices):
        for node in charging_and_queue_totals[sim_index]:
            total_minus_capacity = charging_and_queue_totals[sim_index][node] - sim.station_g.nodes[node]['physical_capacity']
            queue_totals[sim_index][node] = total_minus_capacity if total_minus_capacity>0 else 0
    return queue_totals

def get_vehicles_charging(sim):
    charging_and_queue_totals = get_node_totals(sim)
    charging_totals = {sim_index:{} for sim_index in charging_and_queue_totals}
    sim_length_indices = int(sim.simulation_length/sim.time_interval)
    for sim_index in range(sim_length_indices):
        for node in charging_and_queue_totals[sim_index]:
            total_vehicles = charging_and_queue_totals[sim_index][node]
            physical_capacity = sim.station_g.nodes[node]['physical_capacity']
            if total_vehicles <= physical_capacity:
                charging_totals[sim_index][node] = total_vehicles
            else:
                charging_totals[sim_index][node] = physical_capacity
    return charging_totals

def get_total_vehicles_in_queues(sim):
    queue_lengths = get_node_queue_lengths(sim)
    vehicles_in_queues = {}
    for sim_index in queue_lengths:
        vehicles_in_queues[sim_index] = sum(queue_lengths[sim_index].values())
    return vehicles_in_queues

def get_utilization(sim):
    charging_and_queue_totals = get_node_totals(sim)
    utilization_totals = {}
    sim_length_indices = int(sim.simulation_length/sim.time_interval)
    for sim_index in range(sim_length_indices):
        for node in charging_and_queue_totals[sim_index]:
            if sim_index not in utilization_totals:
                utilization_totals[sim_index]= {}
            else:
                utilization_totals[sim_index][node] = charging_and_queue_totals[sim_index][node]/sim.station_g.nodes[node]['physical_capacity']
    return utilization_totals

##################### aggregate totals #####################
def get_sim_index_counts(sim, loc_type):
    arr = get_loc_array(sim, True)
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            if arr[i][j] == loc_type:
                arr[i][j] = 1
            else:
                arr[i][j] = 0
    return np.sum(np.array(arr).astype('int'), axis = 0)

def get_hour_counts(sim, loc_type):
    steps_per_hour = int(1/sim.time_interval)
    num_hours = int(sim.simulation_index/steps_per_hour)
    arr = np.array(get_sim_index_counts(sim, loc_type))
    out = arr.reshape((num_hours, steps_per_hour)).sum(1)
    return {i:val for i, val in enumerate(out)}

def get_aggregate_node_totals(sim):
    node_totals_per_sim_index = get_node_totals(sim)
    node_totals = None
    for sim_index in node_totals_per_sim_index:
        if sim_index == 0:
            node_totals = node_totals_per_sim_index[sim_index]
        else:
            for node in node_totals_per_sim_index[sim_index]:
                node_totals[node] += node_totals_per_sim_index[sim_index][node]
    return node_totals

def get_node_avgs(sim):
    node_totals = get_aggregate_node_totals(sim)
    node_averages = {node:node_totals[node]/sim.simulation_index for node in node_totals}
    return node_averages

LOCATION_TYPES = ["queue", "charging", "road", "src", "dst"]

def get_hours_on_edge_type(sim, finished_status = "finished"):
    if finished_status=="finished":
        vehicle_list = [v for v in sim.vehicle_list if v.travel_time!=0]
    elif finished_status=="unfinished":
        vehicle_list = [v for v in sim.vehicle_list if v.travel_time==0]
    else:
        vehicle_list = sim.vehicle_list
    
    num_vehicles = len(vehicle_list)
    loc_type_all_vehicles = {loc_type:np.zeros(num_vehicles) for loc_type in LOCATION_TYPES}
    steps_per_hour = int(1/sim.time_interval)
    for i, vehicle in enumerate(vehicle_list):
        edge_freqs = get_edge_type_frequencies(vehicle)
        edge_hours = {key:edge_freqs[key]/steps_per_hour for key in edge_freqs}
        for metric in edge_freqs:
            loc_type_all_vehicles[metric][i] = edge_hours[metric]
    return loc_type_all_vehicles

def get_edge_hour_measurements(sim, edge_type, finished_status):
    metric_vehicle_dict = get_hours_on_edge_type(sim, finished_status)
    arr = np.array(metric_vehicle_dict[edge_type])
    d = {"min": np.min(arr), "med": np.median(arr), "max": np.max(arr), "mean": np.mean(arr), "std": np.std(arr), "sum":np.sum(arr)}
    return d

def get_electricity_usage_per_hour(sim):
    charging_kw = 45
    vehicles_charging_per_hour = get_hour_counts(sim, "charging")
    kwh_usage_per_hour = {key:vehicles_charging_per_hour[key]*charging_kw for key in vehicles_charging_per_hour}
    return kwh_usage_per_hour

##################### metrics #####################

def get_utilization_metric(sim, calc_method = "avg_of_disp"):
    '''avg_of_disp, disp_of_avg'''
    
    if calc_method == "total_disp_of_avg":
        node_averages = get_node_avgs(sim)
        utilization_dict = {node:node_averages[node]/sim.station_g.nodes[node]["physical_capacity"] for node in node_averages}
        utilization = sorted(list(utilization_dict.values()))
        u_lower_20 = np.mean(utilization[:int(np.floor(len(utilization)*1/5))])
        u_upper_20 = np.mean(utilization[int(np.ceil(len(utilization)*4/5)):])
        return u_upper_20-u_lower_20
    else:
        sim_node_utils = get_utilization(sim)
        disp = []
        for sim_index in sim_node_utils:
            utilization = sorted(list(sim_node_utils[sim_index].values()))
            u_lower_20 = np.mean(utilization[:int(np.floor(len(utilization)*1/5))])
            u_upper_20 = np.mean(utilization[int(np.ceil(len(utilization)*4/5)):])
            disp.append(u_upper_20-u_lower_20)
        return np.mean(disp)


def get_percent_delay_metric(sim):
    hours_on_edge_types = get_hours_on_edge_type(sim)
    delay = hours_on_edge_types["queue"]+hours_on_edge_types["charging"]
    baseline = hours_on_edge_types["road"]+hours_on_edge_types["src"]+hours_on_edge_types["dst"]
    percent_delay = (baseline+delay)/baseline
    d = {"min": np.min(percent_delay), 
         "med": np.median(percent_delay), 
         "max": np.max(percent_delay),
         "mean": np.mean(percent_delay),
         "std": np.std(percent_delay)}
    return d

def get_electricity_metric(sim):
    kw_per_mw = 1000
    elec_df = pd.read_csv("data_test/Demand_for_California_(region)_hourly_-_UTC_time.csv", skiprows=5, names=["time", "MWH"])
    elec_df["utc_time"] = pd.to_datetime(elec_df["time"])
    elec_df["local_time"] = elec_df["utc_time"] + pd.Timedelta(hours=-8)
    elec_df_2021 = elec_df[elec_df['local_time'].dt.year==2021]
    july_profile_df = elec_df_2021.groupby([elec_df_2021['local_time'].dt.month, elec_df_2021['local_time'].dt.hour]).mean().loc[7]["MWH"]
    july_profile = np.array(july_profile_df)
    max_demand = 1.1*max(july_profile)
    hour_profile = np.resize(july_profile, sim.simulation_hour_index)
    # upper_bound = 1.1*hour_profile
    kwh_usage_per_hour = get_electricity_usage_per_hour(sim)
    simulated_profile = hour_profile+ np.array(list(kwh_usage_per_hour.values()))/kw_per_mw
    return len(np.where(simulated_profile>max_demand)[0])/sim.simulation_hour_index
