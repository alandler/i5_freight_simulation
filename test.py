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

  
if __name__ == '__main__':
    unittest.main()