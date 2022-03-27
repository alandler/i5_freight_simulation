import networkx as nx
# import numpy as np
import pandas as pd
import random
import sys

# TODO if logic controlling testing or real environment 
stations_df = pd.read_csv("data_test/stations.csv")
distances_df = pd.read_csv("data_test/distances.csv")
distances_df = distances_df[distances_df["Total_TravelTime"]!=0]
elec_df = pd.read_csv("data_test/Demand_for_California_(region)_hourly_-_UTC_time.csv", skiprows=5, names=["time", "MWH"])

def set_random_speed_columns():
    free_flow_speed = distances_df["Total_Kilometers"]/(distances_df["Total_TravelTime"]/60)
    hour_factors = {0: 1, 1:1, 2:1, 3:1, 4:1, 5:.95, 6:.95, 7:.87, 8:.75, 9:.8, 10: .85, 11:.9, 12:.85, 
                    13: .9, 14:.9, 15:.9, 16:.85, 17:.75, 18: .7, 19: .75, 20:.8, 21: .85, 22: .95, 23: 1, 24:1}
    for i in range(24):
        distances_df["speed_"+str(i)] = free_flow_speed*random.gauss(hour_factors[i],.05)

def get_station_G(battery_capacity = 215):
    ''' Return a networkx graph containing an augmented graph of the physical network.
    '''
    # add nodes with IDs, charging_rates, positions
    station_G  = nx.DiGraph()
    for index, row in stations_df.iterrows():
        station_G.add_node(str(row["OID_"]), 
                            charging_rate = row["charging_rate"],
                            pos = (row["longitude"],  row["latitude"]),
                            physical_capacity = row["physical_capacity"])

    # TODO: elevation, avg_speed data
    for index, row in distances_df.iterrows():
        station_G.add_edge(str(row["OriginID"]),str(row["DestinationID"]),
                        weight= row["Total_TravelTime"]/60,
                        time = row["Total_TravelTime"]/60,
                        length= row["Total_Kilometers"],
                        battery_cost = row["Total_Kilometers"]*1.9/battery_capacity*100) # battery cost as a percent of total battery capacity consumed (assuming battery capacity of 215kWh)

    return station_G

def prune_station_G(station_G, max_distance = 500):
    ''' This should take edges that are clearly redundant and prune them
    Start by removing all edges with len over 500 km
    TODO: come up with more refined  pruning methods for local redundancies as well'''
    for edge in station_G.edges:
        if station_G.edges[edge]["length"] >= max_distance:
            station_G.remove_edge(*edge)

def ingest_demand_data():
    ''' TODO: We don't have this data yet '''
    pass

def ingest_avg_speed_data():
    ''' TODO: We don't have this data yet '''
    pass

def apply_avg_speeds():
    '''TODO: either use travel times as given with presumptions about hourly distributions or do something else'''
    
    pass

def ingest_electricity_data():
    ''' Returns 2 arrays of 24, representing hourly MWH demand in CA for the winter and summer.'''
    elec_df["utc_time"] = pd.to_datetime(elec_df["time"])
    elec_df["local_time"] = elec_df["utc_time"] + pd.Timedelta(hours=-8)
    elec_df_2021 = elec_df[elec_df['local_time'].dt.year==2021]
    winter_mwh = list(elec_df_2021.groupby([elec_df_2021['local_time'].dt.month, elec_df_2021['local_time'].dt.hour]).mean().loc[1]["MWH"])
    summer_mwh = list(elec_df_2021.groupby([elec_df_2021['local_time'].dt.month, elec_df_2021['local_time'].dt.hour]).mean().loc[8]["MWH"])
    return (winter_mwh, summer_mwh)

if __name__ == '__main__':
    get_station_G()
    set_random_speed_columns()


