import numpy as np
import math
from utils import *


class NN_Solver:
    def __init__(self, coord, edge_weight=None):
        self.coord = coord
        self.num_node = len(coord)
        self.tour = np.arange(self.num_node)

    def calc_dist(self, u, v):
        return np.linalg.norm(self.coord[u] - self.coord[v])

    def eval_obj(self):
        obj_val = 0
        for i in range(self.num_node):
            obj_val += self.calc_dist(self.tour[i],
                                      self.tour[(i + 1) % self.num_node])
        return obj_val

    def solve(self):
        for i in range(1, self.num_node):
            min_dist = float('inf')
            arg_min_dist = None
            # numpy で高速化する余地あり
            for j in range(i, self.num_node):
                dist = self.calc_dist(self.tour[i - 1], self.tour[j])
                if dist < min_dist:
                    min_dist = dist
                    arg_min_dist = j
            self.tour[i], self.tour[arg_min_dist] = self.tour[arg_min_dist], self.tour[i]

        return self.tour


class Local_Solver:
    def __init__(self, coord, time_limit=10, edge_weight=None):
        self.coord = coord
        self.num_node = len(coord)
        self.tour = NN_Solver(coord).solve()

    def solve():
        pass


if __name__ == '__main__':
    data_name = "gr17"
    get_tsplib(data_name)
    coord, opt_tour = load_tsplib(data_name)
    optimizer = NN_Solver(coord)
    print(optimizer.eval_obj())
    sol = optimizer.solve()
    print(optimizer.eval_obj())
    render(coord, sol)
