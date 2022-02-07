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
            battery_str = re.findall(r".*_(\d*)_.*", node)[0]
            in_or_out = re.findall(r".*_\d*_(.*)", node)[0]
            longitude_shift = -2 if in_or_out == "in" else 2
            latitude = int(battery_str)
        except: # sinks
            node_str = node
            longitude_shift = 0
            latitude = 125
        pos[node] = (node_columns[int(node_str)]+longitude_shift, latitude)
        
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