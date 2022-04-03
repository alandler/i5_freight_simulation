import networkx as nx
import numpy as np
import pandas as pd
import random
import sys
from scipy import spatial
from tqdm import tqdm

# TODO: use this data
elec_df = pd.read_csv("data_test/Demand_for_California_(region)_hourly_-_UTC_time.csv", skiprows=5, names=["time", "MWH"])

def select_dataset(dataset):
    datasets = {
        "test": ["stations.csv", "distances.csv"],
        "wcctci": ["wcctci_stations-updated.csv", "wcctci_coord_distances.csv"],
        "wcctci+parking": [],
        "parking": ["stations.csv", "distances.csv"],
    }
    directory = "data/"
    if dataset == "test":
        directory = "data_test/"

    stations_df = pd.read_csv(directory + datasets[dataset][0])
    distances_df = pd.read_csv(directory + datasets[dataset][1])
    distances_df = distances_df[distances_df["Total_TravelTime"]!=0]
    return stations_df, distances_df

def set_random_speed_columns(distances_df):
    ''' Use rush hour estimates to adjust free flow speeds temporally '''
    free_flow_speed = distances_df["Total_Kilometers"]/(distances_df["Total_TravelTime"]/60)
    hour_factors = {0: 1, 1:1, 2:1, 3:1, 4:1, 5:.95, 6:.95, 7:.87, 8:.75, 9:.8, 10: .85, 11:.9, 12:.85, 
                    13: .9, 14:.9, 15:.9, 16:.85, 17:.75, 18: .7, 19: .75, 20:.8, 21: .85, 22: .95, 23: 1, 24:1}
    for i in range(24):
        distances_df["speed_"+str(i)] = free_flow_speed*random.gauss(hour_factors[i],.05)
    return distances_df

def get_station_g(stations_df, distances_df, battery_capacity = 215):
    ''' Return a networkx graph containing an augmented graph of the physical network.
    '''
    # add nodes with IDs, charging_rates, positions
    station_g  = nx.DiGraph()
    for index, row in stations_df.iterrows():
        station_g.add_node(str(row["OID_"]), 
                            charging_rate = row["charging_rate"],
                            pos = (row["longitude"],  row["latitude"]),
                            physical_capacity = row["physical_capacity"])

    # TODO: elevation, avg_speed data
    for index, row in distances_df.iterrows():
        station_g.add_edge(str(row["OriginID"]),str(row["DestinationID"]),
                        weight= row["Total_TravelTime"]/60,
                        time = row["Total_TravelTime"]/60,
                        length= row["Total_Kilometers"],
                        battery_cost = row["Total_Kilometers"]*1.9/battery_capacity*100) # battery cost as a percent of total battery capacity consumed (assuming battery capacity of 215kWh)

    return station_g

def prune_station_g(station_g, max_distance = 500):
    ''' This should take edges that are clearly redundant and prune them
    Start by removing all edges with len over 500 km
    TODO: come up with more refined  pruning methods for local redundancies as well'''
    for edge in station_g.edges:
        if station_g.edges[edge]["length"] >= max_distance:
            station_g.remove_edge(*edge)

def ingest_demand_data():
    ''' TODO: USE PEMMS'''
    pass

def ingest_avg_speed_data(stations_df, distances_df):
    ''' Ingest PEMS data and extract speeds at road endpoints 
    Requires: longitude, latitude as columns in stations_df'''

    def nearest_speed(x):
        lon = x["longitude"]
        lat = x["latitude"]
        d_lon = x["longitude_dst"]
        d_lat = x["latitude_dst"]
        src_res = tree.query([(lat,lon)])
        dst_res = tree.query([(d_lat,d_lon)])
        src_speed = df_h.iloc[src_res[1][0]]["avg_speed"]
        dst_speed = df_h.iloc[dst_res[1][0]]["avg_speed"]
        return (src_speed+dst_speed)/2

    columns = ['time', 'station', 'district', 'route', 'direction_of_travel', 
           'lane_type', 'station_length', 'samples', 'percent_observed', 
           'total_flow', 'avg_occupancy', 'avg_speed', 'delay_35', 'delay_40', 
           'delay_45', 'delay_50', 'delay_55', 'delay_60']
    meta_csvs = {'3': 'd03_text_meta_2022_03_05.txt',
    '4':'d04_text_meta_2022_03_25.txt',
    '5':'d05_text_meta_2021_03_26.txt',
    '6':'d06_text_meta_2022_03_08.txt',
    '7':'d07_text_meta_2022_03_12.txt',
    '8':'d08_text_meta_2022_03_25.txt',
    '10':'d10_text_meta_2022_02_24.txt',
    '11':'d11_text_meta_2022_03_16.txt',
    '12':'d12_text_meta_2021_05_18.txt'}

    df = pd.DataFrame()

    for i in ['3', '4', '5', '6', '7', '8', '10', '11', '12']:
        num = i
        if int(i) < 10:
            num = "0" + i
        speed_df = speed_df = pd.read_csv("pems_ingest/station_data/d"+num+"_text_station_hour_2022_02.txt", sep = ',', header = None)
        speed_df = speed_df.iloc[: , :len(columns)]
        speed_df = speed_df.rename(columns = {i:columns[i] for i in range(len(columns))})
        speed_df["time"] = pd.to_datetime(speed_df['time'])
        speed_df["hour"] = speed_df["time"].dt.hour
        meta_df = pd.read_csv("pems_ingest/station_data/" + meta_csvs[i], sep = "\t")
        meta_df = meta_df[["ID", "Latitude", "Longitude"]]
        meta_df = meta_df.set_index("ID")
        speed_df = speed_df.join(meta_df, on = "station")
        df = pd.concat([df, speed_df])
        df = df.dropna(axis="index", how="any", subset=["Latitude", "Longitude"])
    
    coord_distances_df = distances_df.join(stations_df.set_index("OID_")[["longitude", "latitude"]], on= "OriginID", rsuffix="_origin")
    coord_distances_df = coord_distances_df.join(stations_df.set_index("OID_")[["longitude", "latitude"]], on= "DestinationID", rsuffix="_dst")

    for h in tqdm(range(24)):
        df_h = df[df["hour"]==h]
        df_h = df_h.groupby(by=['station']).mean()
        df_h = df_h.dropna(axis="index", how="any", subset=["avg_speed", "Latitude", "Longitude"])
        coords = df_h[["Latitude", "Longitude"]].to_numpy()
        tree = spatial.KDTree(coords)
        coord_distances_df["speed_" + str(h)] = coord_distances_df.apply(nearest_speed, 1)

    coord_distances_df.to_csv("coord_distances.csv")

def ingest_electricity_data():
    ''' Returns 2 arrays of 24, representing hourly MWH demand in CA for the winter and summer.'''
    elec_df["utc_time"] = pd.to_datetime(elec_df["time"])
    elec_df["local_time"] = elec_df["utc_time"] + pd.Timedelta(hours=-8)
    elec_df_2021 = elec_df[elec_df['local_time'].dt.year==2021]
    winter_mwh = list(elec_df_2021.groupby([elec_df_2021['local_time'].dt.month, elec_df_2021['local_time'].dt.hour]).mean().loc[1]["MWH"])
    summer_mwh = list(elec_df_2021.groupby([elec_df_2021['local_time'].dt.month, elec_df_2021['local_time'].dt.hour]).mean().loc[8]["MWH"])
    return (winter_mwh, summer_mwh)

if __name__ == '__main__':
    select_dataset("wcctci")


