# Helper functions for working with paths

def distance_segment_path(station_g, path, seg_dis):
    '''Returns a modified station_g including paths segmented into equidistant segments.
    The splits minimize the difference from seg_dis.'''
    for i in range(len(path)-1):
        src = path[i]
        dst = path[i+1]

        # get (or set) segment_length and num_segments
        try:
            segment_length = station_g[src][dst]['segment_length']
            num_segments = station_g[src][dst]['num_segments']
        except:
            total_length = station_g[src][dst]['length']
            num_segments = round(total_length/seg_dis, 0)
            segment_length = total_length/num_segments
            station_g[src][dst]['segment_length'] = segment_length
            station_g[src][dst]['num_segments'] = num_segments

        # remove src-dst connection
        station_g.remove_edge(src, dst)

        current_src = src
        current_dst = dst
        # connect src->dst via bi-directional segments
        for i in range(num_segments):
            if i == num_segments-1:
                current_dst = dst
            else:
                current_dst = src + "_" + dst +"_s" + str(i) # nomenclature: src_dst_s1, src_dst_s2
            station_g.add_edge(current_src,current_dst)
            station_g.add_edge(current_dst, current_src)
            current_src = current_dst

    return station_g

def time_segment_path(G, start_time, path, time_interval, end_simulation_time):
    '''Given a vehicle, list its positions at time_interval markings'''
    
    # initialize 
    simulation_time = 0
    total_path_time = start_time
    segment_locations = []
    
    # record None for simulation time during which vehicle hasn't left
    while start_time>simulation_time:
        segment_locations.append(None)
        simulation_time+=time_interval
    
    # iterate over whole path
    for i in range(len(path)-1):
        #intialize variables
        src = path[i]
        dst = path[i+1]
        edge_time = G.get_edge_data(src, dst)['weight']
        
        # has not started traversing current edge
        curr_edge_time = 0
        
        # if the next simulation interval falls on the edge, record it
        while simulation_time <= total_path_time + edge_time:
            
            # charging: src_in to dst_out
            if src[-3:]=="_in" and src[:-3]!=dst[:-4]:
                charging_node = src.split("_")[0]
                segment_locations.append(charging_node)
            # road: src_out to dst_in
            else:
                src_node = src.split("_")[0]
                dst_node = dst.split("_")[0]
                segment_locations.append((src_node, dst_node))
            
            # increment the simulation interval 
            simulation_time += time_interval
            
        # update total path time after edge is traversed (and recorded, if needed)
        total_path_time += edge_time
    
    # add None's once at destination
    while simulation_time <= end_simulation_time:
        segment_locations.append(None)
        simulation_time+=time_interval
    
    # return the locations and the time to traverse the path (not including empty simulation time)
    return (segment_locations, total_path_time-start_time)
