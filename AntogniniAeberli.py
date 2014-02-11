#!/usr/bin/env python
"""usage: AntogniniAeberli.py [options] [params] [cityfile]

options:
-h, --help Show this help
-n, --no-gui

params:
-m VALUE, --maxtime=VALUE  Max execution time of genetic algorithm.
                           Negative values for infinite. Default: 0

(c) 2014 by Diego Antognini and Marco Aeberli")
"""

import sys
import getopt
import os
import pygame
from pygame.locals import KEYDOWN, QUIT, MOUSEBUTTONDOWN, K_RETURN, K_ESCAPE
from math import sqrt
from random import randint, random
from copy import deepcopy

screen_x = 500
screen_y = 500
city_color = [10, 10, 200]
city_radius = 3
font_color = [255, 255, 255]


class Town:
    def __init__(self, id, name, x, y):
        self.id = id
        self.name = name
        self.x = float(x)
        self.y = float(y)

    @staticmethod
    def compute_distance(t1, t2):
        return sqrt(abs(t1.x-t2.x)**2 + abs(t1.y-t2.y)**2)


class Solution:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness_score = -1.0

    def __repr__(self):
        return str(self.fitness_score) + " : " + " ".join(self.chromosome)

    def __len__(self):
        return len(self.chromosome)

    def __getitem__(self, item):
        return str(self.chromosome[item])

    def __setitem__(self, key, value):
        self.chromosome[key] = str(value)

    def index(self, value):
        return self.chromosome.index(str(value))


class Problem:
    NB_POPULATION = 1000
    MUTATION_RATE = 0.01
    MAX_GENERATION_ALLOWED = 100000
    CROSSOVER_RATE = 0.7

    def __init__(self, cities):
        self.cities = []
        self.cities_dict = {}
        self.nb_char, self.keys = Problem.create_alphabet(cities)
        for c in range(0, len(cities)):
            town = Town(self.keys[c], cities[c][0], cities[c][1], cities[c][2])
            self.cities_dict[town.id] = town
            self.cities.append(town)

        self.best_solution = ""
        self.population = []
        self.create_population()

    def generate(self):
        self.best_solution = Solution([])
        self.best_solution.fitness_score = float('inf')

        for i in range(0, Problem.MAX_GENERATION_ALLOWED):
            fitness_scores_total = 0.0
            old_solution = self.best_solution

            for p in self.population:
                if p.fitness_score < 0:
                    p.fitness_score = self.fitness_score(p)
                fitness_scores_total += p.fitness_score
                if p.fitness_score < self.best_solution.fitness_score:
                    self.best_solution = p

            if old_solution != self.best_solution:
                print(self.best_solution)

            solution1 = self.roulette(fitness_scores_total)
            solution2 = solution1
            while solution2 == solution1:
                solution2 = self.roulette(fitness_scores_total)

            if random() < Problem.CROSSOVER_RATE:
                self.population.append(Problem.crossover(solution1, solution2, self.keys, self.nb_char))
            Problem.mutate(solution1)
            Problem.mutate(solution2)

    def roulette(self, fitness_scores_total):
        fitness_score_goal = random()*fitness_scores_total
        fitness_scores_sum = 0.0
        for p in self.population:
            fitness_scores_sum += p.fitness_score
            if fitness_scores_sum >= fitness_score_goal:
                return p
        return self.population[-1]

    def create_population(self):
        for i in range(0, Problem.NB_POPULATION):
            current = []
            j = 0
            keys = deepcopy(self.keys)
            while j < len(self.keys):
                gene_index = randint(0, len(keys)-1)
                gene = keys[gene_index]
                if gene not in current:
                    current.append(gene)
                    keys.pop(gene_index)
                    j += 1
            self.population.append(Solution(current))

    @staticmethod
    def crossover(ga, gb, cities, nb_char):
        fa, fb = True, True
        n = len(cities)
        town = str(randint(0, n-1)).zfill(nb_char)
        x = ga.index(town)
        y = gb.index(town)
        g = [town]
        while fa or fb:
            x = (x - 1) % n
            y = (y + 1) % n
            if fa:
                if ga[x] not in g:
                    g.insert(0, ga[x])
                else:
                    fa = False
            if fb:
                if gb[y] not in g:
                    g.append(gb[y])
                else:
                    fb = False

        remaining_towns = []
        if len(g) < len(ga):
            while len(g)+len(remaining_towns) != n:
                x = (x - 1) % n
                if ga[x] not in g:
                    remaining_towns.append(ga[x])

            while len(remaining_towns) != 0:
                index = randint(0, len(remaining_towns)-1)
                g.append(remaining_towns[index])
                remaining_towns.pop(index)

        return Solution(g)

    @staticmethod
    def mutate(solution):
        if random() < Problem.MUTATION_RATE:
            gene1 = randint(0, len(solution)-1)
            gene2 = gene1
            while gene2 == gene1:
                gene2 = randint(0, len(solution)-1)
            solution[gene2], solution[gene1] = solution[gene1], solution[gene2]

    def fitness_score(self, solution):
        score = 0.0
        for s in range(0, len(solution)-1):
            score += Town.compute_distance(self.cities_dict[solution[s]], self.cities_dict[solution[s+1]])
        score += Town.compute_distance(self.cities_dict[solution[0]], self.cities_dict[solution[-1]])
        return score

    @staticmethod
    def create_alphabet(cities):
        len_cities = len(cities)
        nb_char = 1
        while len_cities > 1:
            len_cities /= 10
            nb_char += 1
        nb_char -= 1

        return nb_char, [str(i).zfill(nb_char) for i in range(0, len(cities))]


def usage():
    """Prints the module how to usage instructions to the console"
    """
    print(__doc__)


def get_argv_params():
    """Recuperates the arguments from the command line
    """
    opts = []
    try:
        opts = getopt.getopt(
            sys.argv[1:],
            "hnm:",
            ["help", "no-gui", "maxtime="])[0]
    except getopt.GetoptError:
        usage()
        print("Wrong options or params.")
        exit(2)
        
    gui = True
    max_time = 0
    
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            exit()
        elif opt in ("-n", "--no-gui"):
            gui = False
        elif opt in ("-m", "--maxtime"):
            max_time = arg
            
    filename = sys.argv[-1]
    if not os.path.exists(filename) or len(sys.argv) <= 1:
        usage()
        print("invalid city file: %s" % filename)
        exit(2)

    return gui, max_time, filename

def ga_solve(file=None, gui=True, max_time=0):
    cities = []
    if file is None:
        #Display GUI and wait a click button to end up the entry
        pass
    else:
        cities = []
        with open(file, 'r+') as f:
            for l in f.readlines():
                cities.append(l.split())
    problem = Problem(cities)
    problem.generate()

if __name__ == "__main__":
    (GUI, MAX_TIME, FILENAME) = (False, 0, 'data/pb010.txt')#get_argv_params()
    print("args gui: %s maxtime: %s filename: %s" % (GUI, MAX_TIME, FILENAME))
    ga_solve(FILENAME, GUI, MAX_TIME)


