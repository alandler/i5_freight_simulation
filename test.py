import networkx as nx
import numpy as np

from simulation import Simulation
from data import get_station_G

import unittest

test_nodes = ["1", "228", "378", "447", "465"]
  
class DataTest(unittest.TestCase):
  
    # Returns True or False. 
    def test(self):        
        self.assertTrue(True)

class GraphTest(unittest.TestCase):
  
    # Returns True or False. 
    def test(self):        
        self.assertTrue(True)

class SimulationTest(unittest.TestCase):

    def test_add_src(self):
        station_G = get_station_G()
        sim = Simulation(station_G)
        src_node = "1"
        src_dstr = [10,20,30,40,50,60,50,40,30,20,30,40,50,60,50,40,30,20,10,5,0,0,0,0]
        src_dstr_neg = [10,20,30,40,50,-60,50,40,30,20,30,40,50,60,50,40,30,20,10,5,0,0,0,0]
        src_dstr_2 = list(np.ones(24))
        sim.add_src(src_node, src_dstr)

        self.assertIsNotNone(sim.src_dict) # src dict exists
        self.assertEqual(sim.src_dict[src_node], src_dstr) # node has proper list

        sim.add_src(src_node, src_dstr_neg)
        self.assertEqual(sim.src_dict[src_node], src_dstr) # negative doesn't update the list

        sim.add_src(src_node, src_dstr_2)
        self.assertEqual(sim.src_dict[src_node], src_dstr_2) # new calls overwrite existing lists

    def test_add_road_time(self):
        station_G = get_station_G()
        sim = Simulation(station_G)
        src_nodes = ["1_25_out", "1_50_out", "1_75_out", "1_100_out"]
        sim.add_road_time("1", "228", 2.5)
        for src in src_nodes:
            for dst in sim.battery_G.neighbors(src):
                if "_in" in dst:
                    self.assertEqual(sim.battery_G[src][dst]['weight'], sim.station_G["1"]["228"]['time'] + 2.5) # all roads +2.5
                else:
                    self.assertEqual(sim.battery_G[src][dst]['weight'], 0) # to sink is still 0

    def test_add_charger_wait_time(self):
        station_G = get_station_G()
        sim = Simulation(station_G)
        src_nodes = ["1_0_in", "1_25_in", "1_50_in", "1_75_in", "1_100_in"]
        dst_nodes = ["1_0_out", "1_25_out", "1_50_out", "1_75_out", "1_100_out"]
        sim.add_charger_wait_time("1", 2.5)
        for i, src in enumerate(src_nodes):
            for j, dst in enumerate(dst_nodes):
                if j > i:
                    self.assertEqual(sim.battery_G[src][dst]['weight'], sim.battery_G[src][dst]['time'] + 2.5) # all charging +2.5
                if j == i:
                    self.assertEqual(sim.battery_G[src][dst]['weight'], 0) # not stopping remains weight 0


  
if __name__ == '__main__':
    unittest.main(SimulationTest())