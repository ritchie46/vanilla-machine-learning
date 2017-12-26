import numpy as np
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
        self.pop_size = pop_size
        self.cross_rate = cross_rate
        self.mutation_rate = mutation_rate
        # Assumed that length > height
        # product: permutations with replacement.
        self.loc = np.array(list(filter(lambda x: x[1] <= height or 1, product(range(length + 1), repeat=2))))

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
                ss.add_element([self.loc[n1], self.loc[n2]])

            # Placing the supports on the outer nodes, and the point load on the middle node.
            x_range = ss.nodes_range('x')
            length = max(x_range)
            middle_node_id = np.argmin(np.abs(np.array(x_range) - length // 2)) + 1
            max_node_id = np.argmin(np.abs(np.array(x_range) - length)) + 1
            ss.add_support_hinged(1)
            ss.add_support_roll(max_node_id)
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

                fitness_w[i] = 1.0 / w
        score = fitness_w * 10 * (1 / fitness_n) * fitness_l
        fitness_l = normalize(fitness_l) * 2
        fitness_w = normalize(fitness_w) * 10
        fitness_n = normalize(1 / fitness_n)

        return fitness_w * fitness_l * fitness_n, score, fitness_w

    def select(self, fitness):
        i = np.random.choice(np.arange(self.pop_size), size=self.pop_size, replace=True, p=fitness / np.sum(fitness))
        return self.pop[i]

    def crossover(self, parent, pop, fitness):
        if np.random.rand() < self.cross_rate:
            i = np.random.choice(np.arange(self.pop_size), size=1, p=fitness / np.sum(fitness))
            # i = np.random.randint(0, self.pop_size, size=1)
            cross_index = np.random.randint(0, 2, size=self.comb.shape[0]).astype(np.bool)
            parent[cross_index] = pop[i, cross_index]

        return parent

    def mutate(self, child):
        i = np.where(np.random.random(self.comb.shape[0]) < self.mutation_rate)[0]
        child[i] = np.random.randint(0, 2, size=i.shape)
        return child

    def evolve(self, fitness):
        # fitness_ordered = fitness[np.argsort(fitness)]
        pop = self.select(fitness)
        pop_copy = pop.copy()

        for i in range(pop.shape[0]):
            parent = pop[i]
            child = self.crossover(parent, pop_copy, fitness)
            child = self.mutate(child)
            parent[:] = child

        self.pop = pop


def validate_calc(ss):
    try:
        displacement_matrix = ss.solve()
        return not np.any(np.abs(displacement_matrix) > 1e9)
    except np.linalg.LinAlgError:
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

a = DNA(5, 4, 500, cross_rate=0.8, mutation_rate=0.001)
plt.ion()

with open("save.pkl", "rb") as f:
    a = pickle.load(f)
    a.mutation_rate = 0.025
    a.cross_rate= 0.8

for i in range(150):
    fitness, s, w = a.get_fitness()
    a.evolve(fitness)

    index_max = np.argmax(s)
    print("gen", i, "max fitness", s[index_max], "w", w[index_max])

    if i % 10 == 0:
        plt.cla()
        fig = a.builds[index_max].show_structure(show=False)

        plt.pause(0.5)

    if i % 20 == 0:
        with open("save.pkl", "wb") as f:
            pickle.dump(a, f)

