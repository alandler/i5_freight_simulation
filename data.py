import networkx as nx
# import numpy as np
import pandas as pd
import random
import sys

# TODO if logic controlling testing or real environment 
stations_df = pd.read_csv("data_test/stations.csv")
distances_df = pd.read_csv("data_test/distances.csv")
distances_df = distances_df[distances_df["Total_TravelTime"]!=0]

def get_station_G():
    ''' Return a networkx graph containing an augmented graph of the physical network.
    TODO: charging_rate must be updated, currently a random number.'''

    # get coordinate list to add to the nodes
    # coords = tuple(zip(stations_df["longitude"], stations_df["latitude"]))

    # add nodes with IDs, charging_rates, positions
    station_G  = nx.DiGraph()
    for index, row in stations_df.iterrows():
        station_G.add_node(str(row["OID_"]), 
                            charging_rate = random.randrange(5,10), 
                            pos = (row["longitude"],  row["latitude"]),
                            physical_capacity = row["physical_capacity"])

    # TODO: elevation, avg_speed data
    for index, row in distances_df.iterrows():
        station_G.add_edge(str(row["OriginID"]),str(row["DestinationID"]),
                        weight= row["Total_TravelTime"],
                        time = row["Total_TravelTime"],
                        length= row["Total_Kilometers"],
                        battery_cost = row["Total_Kilometers"]**2/row["Total_TravelTime"]/2) # TODO: this is an arbitrary formula

    return station_G

def ingest_demand_data():
    pass

if __name__ == '__main__':
    get_station_G()


