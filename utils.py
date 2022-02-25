import numpy as np
import math
import os
import gzip
import tarfile
import matplotlib.pyplot as plt
import glob
from urllib.request import urlretrieve


def eval_obj(coord, tour):
    obj_val = 0
    for i in range(len(tour)):
        obj_val += calc_dist(coord, tour[i],
                             tour[(i + 1) % len(tour)])
    return obj_val


def calc_dist(coord, u, v):
    """
    とりえあずmetric tspのみ対応
    """
    return np.linalg.norm(coord[u] - coord[v])


def get_whole_tsplib(save_dir="./data"):
    if os.path.exists(save_dir):
        return
    else:
        os.makedirs(save_dir)

    url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz"
    urlretrieve(url, "./data.tar.gz")
    with tarfile.open('./data.tar.gz', 'r:gz') as f:
        f.extractall(path="./data")
    os.remove("./data.tar.gz")

    for filename in glob.glob(os.path.join("./data", "*.gz")):
        with gzip.open(filename, 'r') as f:
            data = f.read()
        with open(filename[:-3], 'wb') as f:
            f.write(data)
        os.remove(filename)


def get_tsplib(data_name: str, save_dir="./data"):
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None

    url = "http://elib.zib.de/pub/mp-testdata/tsp/tsplib/tsp/"
    dataset_name = data_name + ".tsp"
    opt_tour_name = data_name + ".opt.tour"

    dataset_file = "{}/{}".format(save_dir, dataset_name)
    opt_tour_file = "{}/{}".format(save_dir, opt_tour_name)

    if(not os.path.exists(dataset_file)):
        urlretrieve(url + dataset_name, dataset_file)

    if(not os.path.exists(opt_tour_file)):
        urlretrieve(url + opt_tour_name, opt_tour_file)


def load_tsplib(data_name, save_dir="./data"):
    dataset_name = data_name + ".tsp"
    opt_tour_name = data_name + ".opt.tour"

    dataset_file = "{}/{}".format(save_dir, dataset_name)
    opt_tour_file = "{}/{}".format(save_dir, opt_tour_name)

    with open(dataset_file, "r") as f:
        data = f.read().split("\n")

    with open(opt_tour_file, "r") as f:
        opt_data = f.read().split("\n")

    for i, line in enumerate(data):
        if("DIMENSION" in line):
            data_dim = int(line.split()[-1])

        if(line == "NODE_COORD_SECTION"):
            coord = np.array(
                list(map(lambda x: x.split()[1:], data[i + 1:i + 1 + data_dim])), dtype=float)
            break

    for i, line in enumerate(opt_data):
        if(line == "TOUR_SECTION"):
            opt_tour = np.array(opt_data[i + 1: i + 1 + data_dim], dtype=int)

    return coord, opt_tour


def render(data, tour=None):
    data_dim = len(data)
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(data[:, 0], data[:, 1])
    if not tour is None:
        for i in range(data_dim):
            p = data[tour[i] - 1]
            np = data[tour[(i + 1) % data_dim] - 1]
            plt.plot([p[0], np[0]], [p[1], np[1]], c='k', linewidth=3)
    plt.show()


if __name__ == '__main__':
    get_whole_tsplib()
    # data_name = "a280"
    # get_tsplib(data_name)
    # data, opt_tour = load_tsplib(data_name)
    # render(data, opt_tour)
