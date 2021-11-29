import networkx as nx
# import numpy as np
import pandas as pd
import random

stations_df = pd.read_csv("data/stations.csv")
distances_df = pd.read_csv("data/distances.csv")

def get_station_G():

    # get positions, add to nodes
    coords = tuple(zip(stations_df["longitude"], stations_df["latitude"]))

    # add nodes with IDs, charging_rates, pos
    station_G  = nx.Graph()
    station_ID_list = list(stations_df["OID_"])
    for i in range(stations_df.shape[0]):
        station_G.add_node(station_ID_list[i], charging_rate = random.randrange(5,10), pos = coords[i])

    # add edges with weight, length
    # TODO: elevation, avg_speed data
    no_self_distances_df = distances_df[distances_df["Total_TravelTime"]!=0]
    src = list(no_self_distances_df["OriginID"])
    dst = list(no_self_distances_df["DestinationID"])
    weight = list(no_self_distances_df["Total_TravelTime"])
    length = list(no_self_distances_df["Total_Kilometers"])

    for i in range(no_self_distances_df.shape[0]):
        station_G.add_edge(src[i],dst[i],
                        weight=weight[i],
                        time = weight[i],
                        length=length[i],
                        battery_cost = length[i]**2/weight[i]/2)
    
    return station_G


