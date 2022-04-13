import re

def set_draw_attributes(G, station_G):
    ''' each are in rows, with levels _in and _out underneath sinks 
    does NOT handle segementation''' 
    
    # set positions
    node_columns = {node: 7*index for index, node in enumerate(list(station_G.nodes))}
    pos = {x: (0,0) for x in G.nodes}
    for node in G.nodes:
        try: # battery layers
            node_str = re.findall(r"(.*)_\d*_.*", node)[0]
            battery_str = re.findall(r".*_(\d*)_.*", node)[0] # node.split("_")
            in_or_out = re.findall(r".*_\d*_(.*)", node)[0]
            longitude_shift = -2 if in_or_out == "in" else 2
            latitude = int(battery_str)
        except: # sinks
            node_str = node
            longitude_shift = 0
            latitude = 125
        pos[node] = (node_columns[node_str]+longitude_shift, latitude)
        
    # set colors
    edge_colors = {}
    for edge in G.edges:
        if "in" not in edge[0] and "in" not in edge[1]:
            edge_colors[edge] = "tab:gray"
        elif "out" not in edge[0] and "out" not in edge[1]:
            edge_colors[edge] = "tab:gray"
        elif "in" in edge[0] and "out" in edge[1]:
            edge_colors[edge] = "tab:green"
        elif "in" in edge[1] and "out" in edge[0]:
            edge_colors[edge] = "tab:red"
        else:
            edge_colors[edge] = "tab:black"
    return pos, edge_colors

def i5_flows(df):
    ''' df is the first df returned by the ingest_pems() function in data.py '''
    #i5 only
    north_i5 = df[(df["route"]==5) & (df["hour"]==8) & (df["direction_of_travel"]=="N")].groupby(by="station").agg({'avg_speed': 'mean','total_flow': 'mean', "Longitude":"first", "Latitude":"first"}).sort_values(by="Latitude")
    i5n_g = nx.DiGraph()
    last_node = None
    for i in range(len(north_i5)):
        row = north_i5.iloc[i]
        i5n_g.add_node(row.name, avg_speed=row["avg_speed"], total_flow=row["total_flow"], pos = (row['Latitude'], row['Longitude']))
        if last_node != None:
            i5n_g.add_edge(last_node, row.name)
        last_node = row.name
        
    import matplotlib.pyplot as plt
    # create number for each group to allow use of colormap
    from itertools import count
    # get unique groups
    groups = set(nx.get_node_attributes(i5n_g,'total_flow').values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = i5n_g.nodes()
    colors = [mapping[i5n_g.nodes[n]['total_flow']] for n in nodes]

    # drawing nodes and edges separately so we can capture collection for colobar
    pos = nx.get_node_attributes(i5n_g,'pos')
    # ec = nx.draw_networkx_edges(i5n_g, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(i5n_g, pos, nodelist=nodes, node_color=colors, 
                                node_size=10, cmap=plt.cm.jet)
    # plt.figure(1,figsize=(20,20)) 
    plt.colorbar(nc)
    plt.axis('off')
    plt.show()