import numpy as np
import math
import random
#from sympy import evaluate
from utils import *
import time
from scipy.spatial import distance
from collections import deque

class Sweep_Solver:
    def __init__(self, coord, edge_weight=None):
        self.coord = coord
        self.num_node = len(coord)
        self.tour = np.arange(self.num_node)
        self.obj = eval_obj(self.coord, self.tour)
        self.dist_matrix = distance.cdist(coord, coord, metric='euclidean') 

    def solve(self):
        sum_x = 0
        sum_y = 0
        for x, y in self.coord:
            sum_x += x
            sum_y += y
        x_g = sum_x / self.num_node
        y_g = sum_y / self.num_node
        kakudo = {}
        kyori = {}
        for i in range(self.num_node):
            x = self.coord[i][0]
            y = self.coord[i][1]
            kakudo[i] = math.atan2(x, y)
            kyori[i] = math.sqrt((x - x_g) ** 2 + (y-y_g) ** 2)

        kakudo_sorted = sorted(kakudo.items(), key=lambda x:x[1])
        route = deque()
        for i in range(self.num_node):
            route.append(kakudo_sorted[i][0])

        while route[0] != 0:
            route.append(route.popleft())

        self.tour = np.array(route)
        self.obj = eval_obj(self.coord, self.tour)

        return self.tour


class Local_Solver:
    def __init__(self, coord, time_limit=10, edge_weight=None):
        self.coord = coord
        self.num_node = len(coord)
        init_opt = Sweep_Solver(coord)
        self.tour = init_opt.solve()
        self.obj = init_opt.obj
        self.time_limit = time_limit

    def eval_diff_two_opt(self, i, j):
        u, next_u = self.tour[i], self.tour[(i + 1) % self.num_node]
        v, next_v = self.tour[j], self.tour[(j + 1) % self.num_node]
        cur = calc_dist(self.coord, u, next_u) + \
            calc_dist(self.coord, v, next_v)
        new = calc_dist(self.coord, u, v) + \
            calc_dist(self.coord, next_u, next_v)
        return new - cur

    def eval_diff_Or_opt(self, s, i, j):
        head_p, tail_p = self.tour[i], self.tour[(i + s - 1) % self.num_node]
        prev_p, next_p = self.tour[(
            i - 1) % self.num_node], self.tour[(i + s) % self.num_node]
        v, next_v = self.tour[j %
                            self.num_node], self.tour[(j + 1) % self.num_node]
        cur = calc_dist(self.coord, prev_p, head_p) + calc_dist(self.coord,
                                                                tail_p, next_p) + calc_dist(self.coord, v, next_v)
        new = calc_dist(self.coord, prev_p, next_p) + calc_dist(self.coord,
                                                                v, head_p) + calc_dist(self.coord, tail_p, next_v)
        return new - cur

    def change_tour_two_opt(self, i, j):
        self.obj += self.eval_diff_two_opt(i, j)
        self.tour[i + 1:j + 1] = self.tour[i + 1:j + 1][::-1]

    def change_tour_Or_opt(self, s, i, j):
        #	update	objective	value
        self.obj += self.eval_diff_Or_opt(s, i, j)
        #	get	sub-path	[i,...,i+s-1]
        subpath = []
        for h in range(s):
            subpath.append(self.tour[(i + h) % self.num_node])
        #	move	sub-path	[i,...,i+s-1]	to	j+1
        for h in range(i + s, j + 1):
            self.tour[(h - s) % self.num_node] = self.tour[h % self.num_node]
        for h in range(s):
            self.tour[(j + 1 - s + h) % self.num_node] = subpath[h]

    def two_opt(self, strategy="first"):
        """
        strategy : ["first", "best"] # 即時移動 or 最良優先
        """
        restart = True
        while restart:
            restart = False
            nbhd = [(i, j) for i in range(len(self.tour))
                    for j in range(i + 2, len(self.tour))]

            for i, j in nbhd:
                delta = self.eval_diff_two_opt(i, j)
                if delta < 0:
                    self.change_tour_two_opt(i, j)
                    restart = True
                    break

    def Or_opt(self, size=3, strategy="first"):
        nbhd = [(s, i, j)
                for s in range(1, size + 1)
                for i in range(len(self.tour))
                for j in range(i + s, i + len(self.tour) - 1)
                ]
        restart = True
        while restart:
            for s, i, j in nbhd:
                restart = False
                delta = self.eval_diff_Or_opt(s, i, j)
                if delta < 0:
                    self.change_tour_Or_opt(s, i, j)
                    restart = True
                    break

    def solve(self, method="grasp"):
        start_time = cur_time = time.time()

        while(cur_time - start_time < self.time_limit):
            self.two_opt()
            self.Or_opt()
            print(self.obj)
            cur_time = time.time()
        return self.tour


if __name__ == '__main__':
    data_name = "att48"
    coord, opt_tour = load_tsplib(data_name)
    # kruskal solver
    optimizer = Sweep_Solver(coord)
    print(optimizer.obj)
    sol = optimizer.solve()
    print(optimizer.obj)
    render(coord, sol)

    # Local solver
    optimizer = Local_Solver(coord)
    print(optimizer.obj)
    sol = optimizer.solve()
    print(optimizer.obj)
    render(coord, sol)
