import numpy as np
import math
import random
from sympy import evaluate
from utils import *
import time
from scipy.spatial import distance
import itertools


class UnionFind():
    def __init__(self, n):
        self.par = [-1] * n     #par[x]:要素xの親頂点の番号（自身が根の場合は-1）
        self.rank = [0] * n     #rank[x]:要素xが属する根付き木の高さ
        self.siz = [-1] * n    #siz[x]:要素xが属する根付き木に含まれる頂点数
    
    #根を求める
    def root(self, x):
        if self.par[x] == -1:   #xが根の場合はxを返す
            return x
        else:
            self.par[x] = self.root(self.par[x])    #経路圧縮
            return self.par[x]
    
    #xとyが同じグループに属するか（根が一致するか）
    def issame(self, x, y):
        return self.root(x) == self.root(y)
    
    #xを含むグループとyを含むグループを併合する
    def unite(self, x, y):
        #x側とy側の根を取得する
        rx = self.root(x)
        ry = self.root(y)
        if rx == ry:
            return False    #すでに同じグループのときは何もしない
        #union by size
        if self.rank[rx] > self.rank[ry]:   #ry側のrankが小さくなるようにする
            rx, ry = ry, rx
        self.par[ry] = rx   #ryをrxの子とする
        if self.rank[rx] == self.rank[ry]:  #rx側のrankを調整する
            self.rank[rx] += 1
        self.siz[rx] += self.siz[ry]    #rx側のsizを調整する
        return True
    
    #xを含む根付き木のサイズを求める
    def size(self, x):
        return self.siz[self.root(x)]

class kruskal_Solver:
    def __init__(self, coord, edge_weight=None):
        self.coord = coord
        self.num_node = len(coord)
        self.tour = np.arange(self.num_node)
        self.obj = eval_obj(self.coord, self.tour)
        self.dist_matrix = distance.cdist(coord, coord, metric='euclidean') 

    def solve(self):
        uf = UnionFind(self.num_node)
        one_array_dist = list(itertools.chain.from_iterable(self.dist_matrix))  #距離行列を1次元に変換
        relation = [[] for i in range(self.num_node)]

        for k in np.argsort(one_array_dist):    #sortしたインデックスで返す(重みの小さい辺から追加していく)
            i = k % self.num_node   #列番号の取得
            j = int(k / self.num_node)  #列番号の取得
            if not uf.issame(i, j) and len(relation[i]) < 2 and len(relation[j]) < 2: #閉路を作らず，1つの頂点から3本以上辺が出ない
                relation[i].append(j)
                relation[j].append(i)
                uf.unite(i, j)

        def go_next():
            for i in range(len(relation[next_city])):
                if relation[next_city][i] in unvisited_cities:
                    return relation[next_city][i]
            return None

        #閉路になっていないので端と端の頂点をつなげる
        edge = []
        for i in range(self.num_node):
            if len(relation[i]) == 1:
                edge.append(i)
        relation[edge[0]].append(edge[1])
        relation[edge[1]].append(edge[0])
        #順序を求める
        current_city = 0
        unvisited_cities = set(range(1, self.num_node))
        route = [current_city]
        next_city = relation[current_city][0]
        while unvisited_cities:
            unvisited_cities.remove(next_city)
            route.append(next_city)
            next_city = go_next()

        self.tour = np.array(route)
        self.obj = eval_obj(self.coord, self.tour)

        return self.tour


class Local_Solver:
    def __init__(self, coord, time_limit=10, edge_weight=None):
        self.coord = coord
        self.num_node = len(coord)
        init_opt = kruskal_Solver(coord)
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
    optimizer = kruskal_Solver(coord)
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
