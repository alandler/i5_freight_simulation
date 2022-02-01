import networkx as nx
# import numpy as np
import pandas as pd
import random

# take in Natalie's csv's
stations_df = pd.read_csv("data/stations.csv")
distances_df = pd.read_csv("data/distances.csv")

def get_station_G():
    ''' Return a networkx graph containing an augmented graph of the physical network.
    TODO: charging_rate must be updated, currently a random number.'''

    # get coordinate list to add to the nodes
    coords = tuple(zip(stations_df["longitude"], stations_df["latitude"]))

    # add nodes with IDs, charging_rates, positions
    station_G  = nx.Graph()
    station_ID_list = list(stations_df["OID_"])
    for i in range(stations_df.shape[0]):
        station_G.add_node(station_ID_list[i], charging_rate = random.randrange(5,10), pos = coords[i])

    # extract lists of sources, destinations, travel times, and km lengths
    # TODO: elevation, avg_speed data
    no_self_distances_df = distances_df[distances_df["Total_TravelTime"]!=0] # no edges from a node to itself
    src = list(no_self_distances_df["OriginID"])
    dst = list(no_self_distances_df["DestinationID"])
    weight = list(no_self_distances_df["Total_TravelTime"])
    length = list(no_self_distances_df["Total_Kilometers"])

    # Add edges with travel time as weight, then default time, km length, and battery cost
    for i in range(no_self_distances_df.shape[0]):
        station_G.add_edge(src[i],dst[i],
                        weight=weight[i],
                        time = weight[i],
                        length=length[i],
                        battery_cost = length[i]**2/weight[i]/2) # TODO: this is an arbitrary formula
    
    return station_G


