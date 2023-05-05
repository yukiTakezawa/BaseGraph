import sys
sys.path.append("../")
import unittest
from base_graph import *
from tqdm import tqdm
from numpy.testing import assert_array_equal, assert_array_almost_equal


class IsFiniteTimeConvergence(unittest.TestCase):
    def setUp(self):
        pass
        
    def test(self):
        for max_degree in [1,2,3,4]:
            for n_nodes in tqdm(range(1, 100)):
                graphs = BaseGraph(n_nodes, max_degree)

                init_w = torch.eye(n_nodes)
                for w in graphs.w_list:
                    init_w = init_w.matmul(w)
                    
                assert_array_almost_equal(init_w.numpy(), torch.ones((n_nodes, n_nodes))/n_nodes)
                self.assertTrue(len(graphs.w_list) <= 2*math.log(n_nodes, max_degree+1)+2)
                
            print(f"OK: k={max_degree}")
                
if __name__ == "__main__":
    unittest.main()
