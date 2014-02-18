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
from random import randint, random, shuffle
from time import clock


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
    CROSSOVER_FRACTION = 0.7
    MAX_GENERATION_ALLOWED = 10000

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

    @staticmethod
    def create_population(keys):
        population = []
        for i in range(0, Problem.NB_POPULATION):
            current = []
            shuffle(keys)  # Use Fisher-Yates shuffle, O(n). Better than copying and removing
            for k in keys:
                current.append(k)
            population.append(Solution(current))
        return population

    def initialize(self):
        self.best_solution = Solution([])
        self.best_solution.fitness_score = float('inf')
        self.population = self.create_population(self.keys)

    def generate(self):
        fitness_scores_total = 0.0

        for p in self.population:
            p.fitness_score = Problem.fitness_score(p, self.cities_dict)
            fitness_scores_total += p.fitness_score
            if p.fitness_score < self.best_solution.fitness_score:
                self.best_solution = p

        new_population = []
        while len(new_population) != Problem.NB_POPULATION:
            Problem.selection_process(self.population, new_population, fitness_scores_total)
            Problem.crossover_process(self.population, new_population, fitness_scores_total, self.keys, self.nb_char)
            Problem.mutation_process(new_population)
        self.population = new_population

        return self.best_solution

    @staticmethod
    def fitness_score(solution, cities_dict):
        score = 0.0
        for s in range(0, len(solution)-1):
            score += Town.compute_distance(cities_dict[solution[s]], cities_dict[solution[s+1]])
        score += Town.compute_distance(cities_dict[solution[0]], cities_dict[solution[-1]])
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

    @staticmethod
    def selection_process(population_original, new_population, fitness_scores_total):
        population = population_original[:]
        return Problem.select_roulette(population, new_population, fitness_scores_total)

    @staticmethod
    def select_roulette(population, new_population, fitness_scores_total):
        for i in range(0, int((1-Problem.CROSSOVER_FRACTION)*Problem.NB_POPULATION)):
            solution = Problem.roulette(fitness_scores_total, population)
            fitness_scores_total -= population[solution].fitness_score
            new_population.append(population[solution])
            population[solution], population[-1] = population[-1], population[solution]
            population.pop()
        return new_population

    @staticmethod
    def select_tournament(population, new_population):
        """Tournament selection often yields a more diverse population than
        the fitness proportionate selection (roulette wheel). Machine Learning, P256"""
        for i in range(0, int((1-Problem.CROSSOVER_FRACTION)*Problem.NB_POPULATION)):
            key1 = randint(0, len(population)-1)
            key2 = key1
            while key1 == key2:
                key2 = randint(0, len(population)-1)
            solution = Problem.tournament(key1, key2, population)
            new_population.append(population[solution])
            population[solution], population[-1] = population[-1], population[solution]
            population.pop()
        return new_population

    @staticmethod
    def roulette(fitness_scores_total, population):
        fitness_score_goal = random()*fitness_scores_total
        fitness_scores_sum = 0.0
        for p in range(0, len(population)):
            fitness_scores_sum += population[p].fitness_score
            if fitness_scores_sum >= fitness_score_goal:
                return p
        return len(population)-1

    @staticmethod
    def tournament(solution1, solution2, population):
        """Tournament selection often yields a more diverse population than
        the fitness proportionate selection (roulette wheel). Machine Learning, P256"""
        p1 = float(population[solution1].fitness_score)/(population[solution1].fitness_score+population[solution2].fitness_score)
        p2 = float(population[solution2].fitness_score)/(population[solution1].fitness_score+population[solution2].fitness_score)
        p1, p2 = p2, p1  # The shorter result, the better is. We inverse the probability

        return solution1 if random() <= p1 else solution2

    @staticmethod
    def crossover_process(original_population, new_population, fitness_scores_total, keys, nb_char):
        for i in range(0, int(Problem.NB_POPULATION*Problem.CROSSOVER_FRACTION)/2):
            solution1 = original_population[Problem.roulette(fitness_scores_total, original_population)]
            solution2 = solution1
            while solution2 == solution1:
                solution2 = original_population[Problem.roulette(fitness_scores_total, original_population)]
            new_population.append(Problem.crossover(solution1, solution2, keys, nb_char))
            new_population.append(Problem.crossover(solution2, solution1, keys, nb_char))

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
            shuffle(remaining_towns)
            while len(remaining_towns) != 0:
                g.append(remaining_towns.pop())

        return Solution(g)

    @staticmethod
    def mutation_process(new_population):
        nb_mutation = int(Problem.MUTATION_RATE*Problem.NB_POPULATION)
        history = []
        for i in range(0, nb_mutation):
            solution = new_population[randint(0, len(new_population)-1)]
            while solution in history:
                solution = new_population[randint(0, len(new_population)-1)]
            history.append(solution)
            Problem.mutate_swap_town(solution)

    @staticmethod
    def mutate_swap_town(solution):
        gene1 = randint(0, len(solution)-1)
        gene2 = gene1
        while gene2 == gene1:
            gene2 = randint(0, len(solution)-1)
        solution[gene2], solution[gene1] = solution[gene1], solution[gene2]

    @staticmethod
    def mutate_reverse_path(solution):
        gene1 = randint(0, len(solution)-1)
        gene2 = gene1
        while gene2 == gene1:
            gene2 = randint(0, len(solution)-1)
        if gene1 > gene2:
            gene1, gene2 = gene2, gene1
        while gene1 < gene2:
            solution[gene1], solution[gene2] = solution[gene2], solution[gene1]
            gene1 += 1
            gene2 -= 1


class TS_GUI:

    screen_x = 500
    screen_y = 600
    offset_y = 50
    offset_y_between_text = 20
    city_color = [10, 10, 200]
    city_start_color = [255, 0, 0]
    city_radius = 3
    font_color = [255, 255, 255]
    name_cities = 'v'

    def __init__(self):
        pygame.init()
        self.window = pygame.display.set_mode((TS_GUI.screen_x, TS_GUI.screen_y))
        pygame.display.set_caption('Travelling Salesman Problem - Antognini Aeberli')
        self.screen = pygame.display.get_surface()
        self.font = pygame.font.Font(None, 30)
        pygame.display.flip()
        self.cities_dict = {}

    def draw_cities(self):
        self.screen.fill(0)
        i = 0
        for c in self.cities_dict.values():
            self.draw_one_city(int(c.x), int(c.y), TS_GUI.city_start_color if i == 0 else TS_GUI.city_color)
            text = self.font.render("%i cities" % len(self.cities_dict), True, TS_GUI.font_color)
            self.screen.blit(text, (0, TS_GUI.screen_y - TS_GUI.offset_y))
            pygame.display.flip()
            i += 1

    def draw_one_city(self, x, y, color):
        pygame.draw.circle(self.screen, color, (int(x), int(y)), TS_GUI.city_radius)
        pygame.display.flip()

    def draw_path(self, solution, nb_generation):
        self.draw_cities()
        cities_to_draw = []
        for c in range(0, len(solution)):
            town = self.cities_dict[solution[c]]
            cities_to_draw.append((int(town.x), int(town.y)))

        pygame.draw.lines(self.screen, self.city_color, True, cities_to_draw) # True close the polygon between the first and last point
        text = self.font.render("Generation %i, Length %s" % (nb_generation, solution.fitness_score), True, TS_GUI.font_color)
        self.screen.blit(text, (0, TS_GUI.screen_y - TS_GUI.offset_y + TS_GUI.offset_y_between_text))
        pygame.display.flip()

    def read_cities(self):
        running = True
        cities = []
        i = 0
        while running:
            event = pygame.event.wait()
            if event.type == MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                cities.append([TS_GUI.name_cities + str(i), x, y])
                self.draw_one_city(x, y, TS_GUI.city_start_color if i == 0 else TS_GUI.city_color)
                i += 1
            elif event.type == KEYDOWN and event.key == K_RETURN:
                running = False
        return cities

    def display(self, problem, max_time=0):
        old_best_solution = None
        running = True
        i = 0
        t0 = 0
        if max_time > 0:
            t0 = clock()

        while running:
            if i < Problem.MAX_GENERATION_ALLOWED:
                best_solution = problem.generate()
                if old_best_solution != best_solution:
                    old_best_solution = best_solution
                    self.draw_path(old_best_solution, i+1)
                    print("Generation " + str(i + 1) + ":" + str(best_solution))
                i += 1
            event = pygame.event.wait()
            if event.type == QUIT or (max_time > 0 and int(clock()-t0) >= max_time):
                running = False

    def display_text_only(self, problem, max_time=0):
        old_best_solution = None
        t0 = 0
        if max_time > 0:
            t0 = clock()
        for i in range(0, Problem.MAX_GENERATION_ALLOWED):
            best_solution = problem.generate()
            if old_best_solution != best_solution:
                old_best_solution = best_solution
                print("Generation " + str(i + 1) + ":" + str(best_solution))
            if max_time > 0 and int(clock()-t0) >= max_time:
                break

    def quit(self):
        pygame.quit()

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
    g = TS_GUI()
    if file is None:
        cities = g.read_cities()
    else:
        with open(file, 'r+') as f:
            for l in f.readlines():
                cities.append(l.split())

    problem = Problem(cities)
    problem.initialize()

    g.cities_dict = problem.cities_dict

    if gui:
        g.display(problem, max_time)
    else:
        pygame.quit()
        g.display_text_only(problem, max_time)

if __name__ == "__main__":
    (GUI, MAX_TIME, FILENAME) = (False, 0, 'data/pb010.txt')#get_argv_params()
    print("args gui: %s maxtime: %s filename: %s" % (GUI, MAX_TIME, FILENAME))
    ga_solve(FILENAME, GUI, MAX_TIME)


