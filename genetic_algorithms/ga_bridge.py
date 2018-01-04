import numpy as np
import os
import pickle
from scipy.spatial.distance import euclidean
from itertools import combinations, product
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/ritchie46/code/python/anaStruct")
from anastruct.fem.system import SystemElements


class DNA:
    def __init__(self, length, height, pop_size=600, cross_rate=0.8, mutation_rate=0.0001):
        self.length = length
        self.height = height
        self.mirror_line = length // 2
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        # Assumed that length > height
        # product: permutations with replacement.
        self.loc = np.array(list(filter(lambda x: x[1] <= height, product(range(self.mirror_line + 1), repeat=2))))

        # Index tuples of possible connections
        # filters all the vector combinations with an euclidean distance < 1.5.
        # dna
        self.comb = np.array(list(filter(lambda x: euclidean(self.loc[x[1]], self.loc[x[0]]) < 1.5,
                                         combinations(range(len(self.loc)), 2))))

        self.pop = np.random.randint(0, 2, size=(pop_size, len(self.comb)))

        self.builds = None

    def build(self):
        builds = np.zeros(self.pop_size, dtype=object)
        middle_node = np.zeros(self.pop_size, dtype=int)
        all_lengths = np.zeros(self.pop_size, dtype=int)
        n_elements = np.zeros(self.pop_size, dtype=int)

        for i in range(self.pop.shape[0]):
            ss = SystemElements()
            on = np.argwhere(self.pop[i] == 1)

            for j in on.flatten():
                n1, n2 = self.comb[j]
                l1 = self.loc[n1]
                l2 = self.loc[n2]

                ss.add_element([l1, l2])
                # add mirror
                ss.add_element([mirror(l1, self.mirror_line), mirror(l2, self.mirror_line)])

            # Placing the supports on the outer nodes, and the point load on the middle node.
            x_range = ss.nodes_range('x')
            if len(x_range) <= 2:
                builds[i] = None
                all_lengths[i] = 0
                n_elements[i] = 0
            else:
                length = max(x_range)
                start = min(x_range)
                ids = list(ss.node_map.keys())

                max_node_id = ids[np.argmax(x_range)]

                for j in range(self.height):
                    middle_node_id = ss.nearest_node("both", np.array([(length + start) / 2, self.height - j]))
                    if middle_node_id:
                        break

                if middle_node_id is None:
                    middle_node_id = ids[np.argmin(np.abs(np.array(x_range) - (length + start) / 2))]

                ss.add_support_hinged(1)
                ss.add_support_hinged(max_node_id)
                ss.point_load(middle_node_id, Fz=-100)

                builds[i] = ss
                middle_node[i] = middle_node_id
                all_lengths[i] = length
                n_elements[i] = on.size

        self.builds = builds
        return builds, middle_node, all_lengths, n_elements

    def get_fitness(self):
        builds, middle_node, fitness_l, fitness_n = self.build()
        fitness_w = np.zeros(self.pop_size)

        for i in range(builds.shape[0]):
            if validate_calc(builds[i]):

                w = np.abs(builds[i].get_node_displacements(middle_node[i])["uy"])

                x_range = builds[i].nodes_range('x')
                length = max(x_range) - min(x_range)
                fitness_w[i] = 1.0 / (w / ((100 * length**3) / (48 * builds[i].EI)))

        fitness_n = (400 / fitness_n)**2

        return fitness_l**2 + fitness_n + fitness_w, fitness_w, fitness_n

    def crossover(self, parent, pop, fitness):
        if np.random.rand() < self.cross_rate:
            # i = np.random.choice(np.arange(self.pop_size), size=1, p=fitness / np.sum(fitness))
            i = np.random.randint(0, self.pop_size, size=1)
            cross_index = np.random.randint(0, 2, size=self.comb.shape[0]).astype(np.bool)
            parent[cross_index] = pop[i, cross_index]

        return parent

    def mutate(self, child):
        i = np.where(np.random.random(self.comb.shape[0]) < self.mutation_rate)[0]
        child[i] = np.random.randint(0, 2, size=i.shape)
        return child

    def evolve(self, fitness):
        pop = rank_selection(self.pop, fitness)
        pop_copy = pop.copy()

        for i in range(pop.shape[0]):
            parent = pop[i]
            child = self.crossover(parent, pop_copy, fitness)
            child = self.mutate(child)
            parent[:] = child

        self.pop = pop


def rank_selection(pop, fitness):
    order = np.argsort(fitness)[::-1]
    pop = pop[order]

    rank_p = 1 / np.arange(1, pop.shape[0] + 1)
    idx = np.random.choice(np.arange(pop.shape[0]), size=pop.shape[0], replace=True, p=rank_p / np.sum(rank_p))
    return pop[idx]


def validate_calc(ss):
    try:
        a = ss.validate()
        displacement_matrix = ss.solve()
        return not np.any(np.abs(displacement_matrix) > 1e9) and a
    except (np.linalg.LinAlgError, AttributeError):
        return False


def normalize(x):
    if np.allclose(x, x[0]):
        return np.ones(x.shape)*0.1
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def choose_fit_parent(pop):
    """
    https://www.electricmonk.nl/log/2011/09/28/evolutionary-algorithm-evolving-hello-world/

    :param pop: population sorted by fitness
    :return:
    """
    # product uniform distribution
    i = int(np.random.random() * np.random.random() * (pop.shape[1] - 1))
    return pop[i]


def mirror(v, m_x):
    """

    :param v: (array) vertex
    :param m_x: (int) mirror x value
    :return: (array) vertex
    """

    return np.array([m_x + m_x - v[0], v[1]])


a = DNA(10, 6, 200, cross_rate=0.8, mutation_rate=0.05)
# plt.ion()


base_dir = "/home/ritchie46/code/machine_learning/vanilla-machine-learning/genetic_algorithms/img/"
name = "n3"
os.makedirs(os.path.join(base_dir, f"best_{name}"), exist_ok=1)

with open(os.path.join(base_dir, f"best_{name}", "save.pkl"), "rb") as f:
    a = pickle.load(f)
    # a.mutation_rate = 0.1
    # a.cross_rate= 0.8
    f, w, n = a.get_fitness()
    f[np.argwhere(w == 0)] = 0
    idx = np.argmax(f)
    print(w[idx], n[idx])
    a.builds[idx].show_bending_moment()

last_fitness = 0

for i in range(100, 150):
    fitness, w, n = a.get_fitness()

    fitness[np.argwhere(w == 0)] = 0

    a.evolve(fitness)

    max_idx = np.argmax(fitness)
    print("gen", i, "max fitness", fitness[max_idx], "w", w[max_idx], "n", n[max_idx])

    if i % 1 == 0:

        plt.cla()

        if last_fitness != fitness[max_idx]:
            try:
                fig = a.builds[max_idx].show_structure(show=False, verbosity=1)
                plt.title(f"fitness = {round(fitness[max_idx], 3)}")
                fig.savefig(os.path.join(base_dir, f"best_{name}", f"ga{i}.png"))
            except AttributeError:
                pass

        last_fitness = fitness[max_idx]
        # plt.pause(0.5)

    if i % 1 == 0:
        with open(os.path.join(base_dir, f"best_{name}", "save.pkl"), "wb") as f:
            pickle.dump(a, f)

