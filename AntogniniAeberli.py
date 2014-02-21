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
from math import sqrt
from random import randint, shuffle
from time import clock
from copy import deepcopy


def equal_double(a, b, epsilon=1e-6):
    return abs(a-b) < epsilon


class Town:
    """
    Class which represents a town in the TSP.
    """
    def __init__(self, id, name, x, y):
        self.id = id
        self.name = name
        self.x = float(x)
        self.y = float(y)

    @staticmethod
    def compute_distance(t1, t2):
        return sqrt((t1.x-t2.x)**2 + (t1.y-t2.y)**2)


class Solution:
    """
    Class which represents a solution in the TSP. Each gene is represented by an unique id, related with the town.
    """
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.distance = 0

    def __repr__(self):
        return str(self.distance) + " : " + " ".join([str(i) for i in self.chromosome])

    def __len__(self):
        return len(self.chromosome)

    def __getitem__(self, item):
        return self.chromosome[item]

    def __setitem__(self, key, value):
        self.chromosome[key] = value

    def index(self, value):
        return self.chromosome.index(value)


class Problem:
    """
    Class which represents the entire problem (with no gui) for the TSP.
    """
    NB_POPULATION = 0  # Will be changed during the execution time, by FACTOR*len(cities)
    FACTOR = 10  # ~10x number of cities
    SIZE_TOURNAMENT_BATTLE = 20  # Size of the tournament battle with which we keep the best
    MUTATION_RATE = 0.1  # Probability to mutate
    CROSSOVER_FRACTION = 0.7  # Number of generated offsprings
    DELTA_GENERATION = 50  # Convergence criteria. If the best solution hasn't changed since DELTA_GENERATION => STOP

    def __init__(self, cities):
        Problem.NB_POPULATION = len(cities)*Problem.FACTOR
        self.cities = []
        self.cities_dict = {}
        self.keys = Problem.create_alphabet(cities)
        self.best_solution = ""
        self.population = []

        for c in xrange(0, len(cities)):
            town = Town(self.keys[c], cities[c][0], cities[c][1], cities[c][2])
            self.cities_dict[town.id] = town
            self.cities.append(town)

    @staticmethod
    def create_population(keys):
        population = []
        for i in xrange(0, Problem.NB_POPULATION):
            current = []
            shuffle(keys)  # Use Fisher-Yates shuffle, O(n). Better than copying and removing
            for k in keys:
                current.append(k)
            population.append(Solution(current))
        return population

    def initialize(self):
        self.best_solution = Solution([])
        self.best_solution.distance = float('inf')
        self.population = self.create_population(self.keys)
        self.compute_all_distances()

    def compute_all_distances(self):
        for p in self.population:
            Problem.compute_distance(p, self.cities_dict)
            if p.distance < self.best_solution.distance and not equal_double(p.distance, self.best_solution.distance):
                self.best_solution = deepcopy(p)

    def generate(self):
        new_population = []
        Problem.selection_process(self.population, new_population)
        Problem.crossover_process(new_population, self.keys)
        Problem.mutation_process(new_population)
        self.population = new_population
        self.compute_all_distances()

        return self.best_solution

    @staticmethod
    def compute_distance(solution, cities_dict):
        score = 0.0
        for s in xrange(0, len(solution)-1):
            score += Town.compute_distance(cities_dict[solution[s]], cities_dict[solution[s+1]])
        score += Town.compute_distance(cities_dict[solution[0]], cities_dict[solution[-1]])
        solution.distance = score

    @staticmethod
    def create_alphabet(cities):
        return range(0, len(cities))

    @staticmethod
    def selection_process(population, new_population):
        for i in xrange(0, int(round((1-Problem.CROSSOVER_FRACTION)*Problem.NB_POPULATION))):
            indices = set()
            for j in xrange(0, Problem.SIZE_TOURNAMENT_BATTLE):
                k = randint(0, len(population)-1)
                while k in indices:  # We want that indices is composed of unique id
                    k = randint(0, len(population)-1)
                indices.add(k)
            winner = sorted(indices, key=lambda k: population[k].distance)[0]
            # Tricks to pass O(n) to O(1) (worst case)
            population[winner], population[-1] = population[-1], population[winner]
            new_population.append(population.pop())

    @staticmethod
    def crossover_process(new_population, keys):
        future_solution = []
        for i in xrange(0, int(round(Problem.NB_POPULATION*Problem.CROSSOVER_FRACTION)/2)):
            solution1 = new_population[randint(0, len(new_population)-1)]
            solution2 = solution1
            while solution2 == solution1: # We want 2 differents solutions
                solution2 = new_population[randint(0, len(new_population)-1)]
            Problem.run_crossover_2opt(future_solution, solution1, solution2, keys)
        new_population += future_solution

    @staticmethod
    def run_crossover_2opt(new_population, solution1, solution2,  keys):
        new_population.append(Problem.crossover(solution1, solution2, keys))
        new_population.append(Problem.crossover(solution2, solution1, keys))

    @staticmethod
    def run_crossover_ox(new_population, solution1, solution2):
        gene1 = randint(1, len(solution1)-2)
        gene2 = gene1
        while gene2 == gene1:
            gene2 = randint(1, len(solution1)-2)
        if gene1 > gene2:
            gene1, gene2 = gene2, gene1
        new_population.append(Problem.crossover_ox(solution1, solution2[gene1:gene2+1], gene1, gene2))
        new_population.append(Problem.crossover_ox(solution2, solution1[gene1:gene2+1], gene1, gene2))

    @staticmethod
    def crossover_ox(solution_to_copy, cities, p1, p2):
        solution = deepcopy(solution_to_copy)
        for c in cities:
            solution[solution.index(c)] = None

        i = p2
        while i != p1:
            if solution[i] is None:
                j = i + 1
                while j >= len(solution) - 1 or solution[j] is None:
                    if j >= len(solution) - 1:
                        j = 0
                    else:
                        j += 1
                solution[i] = solution[j]
                solution[j] = None
            if i >= len(solution) - 1:
                i = 0
            else:
                i += 1

        for c in cities:
            solution[p1] = c
            p1 += 1
        return solution

    @staticmethod
    def crossover(ga, gb, cities):
        """
        Fore more information, refer to "A Fast TSP Solver Using GA For Java"
        """
        fa, fb = True, True
        n = len(cities)
        town = randint(0, n-1)
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
            shuffle(remaining_towns)  # Use Fisher-Yates shuffle, O(n). Better than copying and removing
            while len(remaining_towns) != 0:
                g.append(remaining_towns.pop())

        return Solution(g)

    @staticmethod
    def mutation_process(new_population):
        nb_mutation = int(round(Problem.MUTATION_RATE*Problem.NB_POPULATION))
        history = []
        for i in xrange(0, nb_mutation):
            solution = new_population[randint(0, len(new_population)-1)]
            while solution in history:
                solution = new_population[randint(0, len(new_population)-1)]
            history.append(solution)
            Problem.mutate_reverse_path(solution)

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
    """
    Class attached with Problem to represent the TSP.
    """
    screen_x = 500
    screen_y = 600
    offset_y = 50
    offset_y_between_text = 20
    offset_x_y_city_name = 10

    city_color = [10, 10, 200]
    city_start_color = [255, 0, 0]
    city_end_color = [0, 255, 0]
    city_radius = 3
    cities_name = 'v'

    infobox_color = [128, 128, 128]
    font_color = [255, 255, 255]

    def __init__(self, gui=True):
        if gui:
            pygame.init()
            self.window = pygame.display.set_mode((TS_GUI.screen_x, TS_GUI.screen_y))
            pygame.display.set_caption('Travelling Salesman Problem - Antognini Aeberli')
            self.screen = pygame.display.get_surface()
            self.font = pygame.font.Font(None, 18)
            self.font_city_name = pygame.font.Font(None, 12)
            pygame.display.flip()
            self.cities_dict = {}

    def draw_one_city(self, name, x, y, color, color_font):
        pygame.draw.circle(self.screen, color, (int(x), int(y)), TS_GUI.city_radius)
        text = self.font_city_name.render(name, True, color_font)
        self.screen.blit(text, (x-TS_GUI.offset_x_y_city_name, y-TS_GUI.offset_x_y_city_name))

    def draw_path(self, solution, nb_generation):
        self.screen.fill(0)
        cities_to_draw = []
        for c in xrange(0, len(solution)):
            color, color_font = TS_GUI.city_color, TS_GUI.font_color
            if c == 0:
                color, color_font = TS_GUI.city_start_color, TS_GUI.city_start_color
            elif c == len(solution)-1:
                color, color_font = TS_GUI.city_end_color, TS_GUI.city_end_color

            town = self.cities_dict[solution[c]]
            self.draw_one_city(town.name, town.x, town.y, color, color_font)
            cities_to_draw.append((int(town.x), int(town.y)))

        pygame.draw.lines(self.screen, self.city_color, True, cities_to_draw)  # True close the polygon between the first and last point

        self.draw_infobox()

        text = self.font.render("Generation %i, Length %s" % (nb_generation, solution.distance), True, TS_GUI.font_color)
        self.screen.blit(text, (0, TS_GUI.screen_y - TS_GUI.offset_y + TS_GUI.offset_y_between_text))

        text = self.font.render("%i cities" % len(self.cities_dict), True, TS_GUI.font_color)
        self.screen.blit(text, (0, TS_GUI.screen_y - TS_GUI.offset_y))

        pygame.display.flip()

    def draw_infobox(self):
        pygame.draw.rect(self.screen, TS_GUI.infobox_color, (0, TS_GUI.screen_y-TS_GUI.offset_y, TS_GUI.screen_x, TS_GUI.offset_y))

    def read_cities(self):
        self.draw_infobox()
        text = self.font.render("Click with the mouse to create a city. Press Enter to continue.", True, TS_GUI.font_color)
        self.screen.blit(text, (0, TS_GUI.screen_y - TS_GUI.offset_y + TS_GUI.offset_y_between_text))
        pygame.display.flip()

        running = True
        cities = []
        i = 0
        while running:
            event = pygame.event.wait()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if y <= TS_GUI.screen_y-TS_GUI.offset_y:
                    cities.append([TS_GUI.cities_name + str(i), x, y])
                    self.draw_one_city(TS_GUI.cities_name + str(i), x, y, TS_GUI.city_color, TS_GUI.font_color)
                    pygame.display.flip()
                    i += 1
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                running = False
        return cities

    def display(self, problem, max_time=0):
        old_best_solution = problem.best_solution
        print("Generation 0 : " + str(old_best_solution))
        self.draw_path(old_best_solution, 0)

        running = True
        i = 1
        t0 = 0
        ith_best = 0

        if max_time > 0:
            t0 = clock()

        while running:
            best_solution = problem.generate()
            if not equal_double(old_best_solution.distance, best_solution.distance):
                old_best_solution = best_solution
                self.draw_path(old_best_solution, i)
                print("Generation " + str(i) + " : " + str(best_solution))
                ith_best = i
            i += 1
            
            event = pygame.event.poll()
            if event.type == pygame.QUIT or (max_time > 0 and int(clock()-t0) >= max_time) or i-ith_best > Problem.DELTA_GENERATION:
                running = False
        return self.return_solution(problem.best_solution)

    def display_text_only(self, problem, max_time=0):
        old_best_solution = problem.best_solution
        print("Generation 0 : " + str(old_best_solution))

        t0 = 0
        i = 1
        ith_best = 0

        if max_time > 0:
            t0 = clock()

        while i-ith_best <= Problem.DELTA_GENERATION and (max_time <= 0 or int(clock()-t0) < max_time):
            best_solution = problem.generate()
            if not equal_double(old_best_solution.distance, best_solution.distance):
                old_best_solution = best_solution
                print("Generation " + str(i) + " : " + str(best_solution))
                ith_best = i
                i += 1
        return self.return_solution(problem.best_solution)

    def return_solution(self, solution):
        cities = []
        for c in xrange(0, len(solution)):
            cities.append(self.cities_dict[solution[c]].name)
        return solution.distance, cities

    def quit(self):
        pygame.quit()


def usage():
    """
    Prints the module how to usage instructions to the console"
    """
    print(__doc__)


def get_argv_params():
    """
    Recuperates the arguments from the command line
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
            max_time = int(arg)
            
    filename = None
    if len(sys.argv) > 1 and os.path.exists(sys.argv[-1]):
        filename = sys.argv[-1];

    return gui, max_time, filename


def ga_solve(file=None, gui=True, max_time=0):
    cities = []
    g = None
    if file is None:
        g = TS_GUI()
        cities = g.read_cities()
        if not gui:
            pygame.quit()
    else:
        with open(file, 'r+') as f:
            for l in f.readlines():
                cities.append(l.split())

    problem = Problem(cities)
    problem.initialize()
    if g is None:
        g = TS_GUI(gui)
    g.cities_dict = problem.cities_dict

    if gui:
        return g.display(problem, max_time)
    else:
        return g.display_text_only(problem, max_time)

if __name__ == "__main__":
    (GUI, MAX_TIME, FILENAME) = get_argv_params()
    print("args gui: %s maxtime: %s filename: %s" % (GUI, MAX_TIME, FILENAME))
    print(ga_solve(FILENAME, GUI, MAX_TIME))


