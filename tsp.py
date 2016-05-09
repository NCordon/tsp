
# coding: utf-8

# # Travelling Salesman Problem

# In[48]:

get_ipython().magic(u'matplotlib inline')


# In[76]:

from numpy import *
from random import *
from math import exp
from copy import deepcopy
from itertools import cycle
import matplotlib as matplotlib
import re


# Definimos una solución como una permutación con su coste

# In[115]:

class Route:
        
        def __init__(self, permutation, dist):
            self.permutation = deepcopy(permutation)
            self.dist = dist
            self.update_cost()

        """Calculates cuadratic cost"""
        def update_cost(self):
            pairs = self.get_edges()
            self.cost = sum([self.dist[x,y] for (x,y) in pairs])
        
        def change_positions(self,i,j):
            # Intercambia dos ciudades del grafo
            self.permutation[i], self.permutation[j] = self.permutation[j], self.permutation[i]
            self.update_cost()
            
        def change_edges(self,i,j):
            # Intercambia dos aristas del grafo
            i,j = min(i,j), max(i,j)
            
            rev = self.permutation[i:j]
            rev = rev[::-1]
            self.permutation[i:j] = rev
            self.update_cost()
        
        def get_edges(self):
            shifted = append(self.permutation[1:], [self.permutation[0]])
            pairs = zip(self.permutation, shifted)
            return(pairs)
            


# Implementación de la clase que albergará los datos del problema

# In[185]:

class TSP:   
    
    #def prueba(self):
    #    return(Route(array(range(len(self.points))), self.dist))
    
    def __read (self, file):  
        f = open(file, 'r')
        match = '^[0-9].*'
        points = []
        
        for line in f:
            is_point = re.search(match, line)

            if is_point:
                x,y = map(float, line.split()[1:])
                points.append((x,y))
    
        return(points)
    
    
    def simulated_annealing(self, t_ini, max_iter, alpha):
        """Temperatura"""
        t = t_ini
        
        """Número de ciudades"""
        n = len(self.points)
        
        """Permutación"""
        permutation = array(range(n))
        shuffle(permutation)
        solution = Route(permutation, self.dist)
        best_solution = Route(permutation, self.dist)
        
        """Variables que controlan las iteraciones"""
        improvement = True
        i=0
        
        while i < max_iter:
            candidate = deepcopy(solution)
            
            # Generamos los índices de los arcos a cambiar
            u = randint(0, n-1)
            v = randint(u+1, n)   
            candidate.change_edges(u,v)
            
            """u = randint(0, n-1)
            v = randint(0, n-1)
            candidate.change_positions(u,v)"""
            
            diff_cost = candidate.cost - solution.cost
            
            if (diff_cost < 0 or random() < exp(-diff_cost*1.0/t)):
                solution = deepcopy(candidate)
        
                if (solution.cost < best_solution.cost):
                    best_solution = deepcopy(solution)
            
            """Esquema de enfriamiento"""
            t = alpha*t
            
            #if (i%100==0):
            #    print(t)
            i+=1
        
        return best_solution
    
    def tabu_search(self, max_iter, max_vecinos, coef_tenencia, aspiration_tol):
        """Número de ciudades"""
        n = len(self.points)
        
        """Permutación"""
        permutation = array(range(n))
        shuffle(permutation)
        candidate = Route(permutation, self.dist)
        best_solution = deepcopy(candidate)
        
        """Lista tabú de soluciones"""
        tabu_list = [None] * int(coef_tenencia*n)
        index = cycle(range(len(tabu_list)))
        
        i = 0
        
        while i < max_iter:
            j = 0
            edge_freq = array([[0]*n]*n)
            
            best_neighbour = None
            entra = False
            while j < max_vecinos: 
                # Generamos los índices de los arcos a cambiar
                u = randint(0, n-1)
                v = randint(u+1, n)   
                candidate.change_edges(u,v)
                
                eval_solution = True

                
                # Si hay arcos comunes entre ambos
                if (set(candidate.get_edges()) & set(tabu_list)):
                    eval_solution = False

                    """Criterio de aspiración"""
                    if best_neighbour is not None and                        candidate.cost < aspiration_tol*best_solution.cost:
                        
                        eval_solution = True

                if eval_solution:
                    if best_neighbour is None: 
                        best_neighbour = deepcopy(candidate)
                    else:
                        if candidate.cost < best_neighbour.cost:
                            best_neighbour = deepcopy(candidate)
                        
                        """Actualizamos los arcos 'buenos'"""
                        for (a,b) in candidate.get_edges():
                            edge_freq[a,b] += 1;
                            entra = True
                        
                        if candidate.cost < best_solution.cost:
                            best_solution = deepcopy(candidate)
                  
                j+=1
                i+=1
            """Fin del while"""
            
            if best_neighbour is not None:
                candidate = deepcopy(best_neighbour)
            # Escoge un elemento aleatorio de los peores arcos visitados por las
            # soluciones y lo introduce en la lista
            used_edges = edge_freq[edge_freq>0]
            
            if used_edges.any():
                tabu_list[ next(index) ] = tuple(choice( 
                    transpose( where(edge_freq == used_edges.min()) )
                ))
            else:
                tabu_list[ next(index) ] = None
            
            #print(tabu_list)
            
        """Fin del while"""
        
        return best_solution
    
        
    def print_solution(self, solution):
        p_x = [ self.points[i][0] for i in solution.permutation ]
        p_y = [ self.points[i][1] for i in solution.permutation ]
        p_x = append(p_x, p_x[0])
        p_y = append(p_y, p_y[0])
        tol_x = 0.05 * mean(p_x)
        tol_y = 0.05 * mean(p_y)
        
        matplotlib.rcParams.update({'font.size': 18, 'lines.linewidth':3})
        matplotlib.pyplot.figure(figsize=(15,10))
        matplotlib.pyplot.xlim(min(p_x) - tol_x, max(p_x) + tol_x)
        matplotlib.pyplot.ylim(min(p_y) - tol_y, max(p_y) + tol_y)
        matplotlib.pyplot.plot(p_x, p_y, marker='o', color='red', markersize=7)
    
    def __init__(self, file):
        self.points = array(self.__read(file))
        self.dist = sqrt(
            [
                [dot(subtract(x,y),subtract(x,y)) for x in self.points] 
                for y in self.points
            ])


# In[175]:

files = ['berlin52.tsp', 'ch150.tsp', 'd198.tsp', 'eil101.tsp']

problems = {}
sa_solutions = {}
ts_solutions = {}
best_solutions = {'berlin52': 7542,
                  'ch150': 6528,
                  'd198': 15780,
                  'eil101': 629}


# In[182]:

semilla = 12345678

for f in files:
    seed(semilla)
    name = f[:-4]
    problems[name] = TSP(f)
    size = len(problems[name].points)
    n_iter = size*100
    alpha = 0.95
    sa_solutions[name] = problems[name].simulated_annealing(size*1e3, n_iter, alpha) 


# In[207]:

semilla = 12345678
f = 'eil101.tsp'

seed(semilla)
name = f[:-4]
problems[name] = TSP(f)
size = len(problems[name].points)
n_iter = size*200
coef_tenencia = 0.3
aspiration_tol = 1.5
ts_solutions[name] = problems[name].tabu_search(n_iter, size, coef_tenencia, aspiration_tol)
ts_solutions[name].cost
problems[name].print_solution(ts_solutions[name])


# In[68]:

for name in problems:
    print (name 
           + '\n\t SA: '   + str(sa_solutions[name].cost)
           + '\n\t Best: ' + str(best_solutions[name]))
    problems[name].print_solution(sa_solutions[name])

