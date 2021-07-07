import math
import random
import visualize_tsp
import matplotlib.pyplot as plt


class SimulatedAnnealing(object):
    def __init__(self, coords, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1,num_it = -1):
        self.coords = coords
        self.N = len(coords)
        #Definisco la temperatura iniziale di default se non è specificata nella chiamata
        self.T = self.N if T == -1 else T
        #Definisco il parametro alpha di default.
        self.alpha = 0.995 if alpha == -1 else alpha
        #Definisco la temperatura finale di default
        self.stopping_temperature = 1e-8 if stopping_T == -1 else stopping_T
        #Definisco il numero di iterazioni per ogni livello di temperatura
        self.num_it = self.N if num_it == -1 else num_it
        #Definisco il numero di iterazioni massimo
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1
        #Inizializzo i nodi
        self.nodes = [i for i in range(self.N)]
        #Inizializzo la soluzione migliore
        self.best_solution = None
        #Inizializzo il valore della funzione obiettivo ad Infinito
        self.best_fitness = float("Inf")
        #Inizializzo l' array di valori della funzione obiettivo
        self.fitness_list = []

    def initial_solution(self):
        """
        Algoritmo greedy per calcolare la soluzione di partenza
        """
        #Imposto come punto di partenza dell' algoritmo greedy il primo nodo
        cur_node = 0
        #Aggiungo il primo nodo all' array soluzione
        solution = [cur_node]
        
        free_nodes = set(self.nodes)
        #Rimuovo il nodo corrente dall' array di nodi
        free_nodes.remove(cur_node)
        while free_nodes:
            #calcolo l' argomento del minimo della distanza tra il nodo corrente e gli altri nodi restanti in free nodes
            next_node = min(free_nodes, key=lambda x: self.dist(cur_node, x))
            #Rimuovo il nodo con distanza minore dal nodo corrente da free_nodes e lo imposto come nuovo nodo corrente
            free_nodes.remove(next_node)
            solution.append(next_node)
            cur_node = next_node
        #Chiamo la funzione fitness che mi calcola il valore della soluzione
        cur_fit = self.fitness(solution)
        #Se la soluzione è migliore di quella trovata fino a questo momento aggiorno la soluzione migliore
        if cur_fit < self.best_fitness: 
            self.best_fitness = cur_fit
            self.best_solution = solution
        self.visualize_routes()
        self.fitness_list.append(cur_fit)
        print("Soluzione iniziale: ",self.fitness(solution))
        
        return solution, cur_fit

    def dist(self, node_0, node_1):
        """
        Calcolo della distanza euclidea tra due punti
        """
        #Estraggo le coordinate del nodo in ingresso e calcolo la distanza euclidea
        coord_0, coord_1 = self.coords[node_0], self.coords[node_1]
        return math.sqrt((coord_0[0] - coord_1[0]) ** 2 + (coord_0[1] - coord_1[1]) ** 2)

    def fitness(self, solution):
        """
        Calcolo del valore della soluzione in ingresso
        """
        cur_fit = 0
        #Calcolo la distanza tra il nodo i e il nodo i+1 per ogni nodo del mio insieme
        for i in range(self.N):
            cur_fit += self.dist(solution[i % self.N], solution[(i + 1) % self.N])
        return cur_fit

    def p_accept(self, candidate_fitness):
        """
        Calcolo della probabilità di accettazione
        """
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)

    def accept(self, candidate):
        """
        Il candidato viene accettato con probabilità 1 se il suo valore di fitness è minore di quello corrente
        Se il suo valore è maggiore di quello corrente il candidato viene accettato con probabilità p_accept
        """
        candidate_fitness = self.fitness(candidate)
        if candidate_fitness < self.cur_fitness:
            self.cur_fitness, self.cur_solution = candidate_fitness, candidate
            if candidate_fitness < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_fitness, candidate
        else:
            #Genero un numero random tra 0 e 1 e lo confronto con il ritorno della funzione p_accept
            if random.random() < self.p_accept(candidate_fitness):
                self.cur_fitness, self.cur_solution = candidate_fitness, candidate

    def anneal(self):
        """
        Esecuzione del corpo del simulated annealing
        """
        # Calcolo della soluzione di partenza con un euristica greedy
        print("Inizio dell' euristica greedy")
        self.cur_solution, self.cur_fitness = self.initial_solution()

        print("Inizio del simulated annealing")
        #Verifica che la temperatura non è minore della temperatura di arresto
        #e che il numero di iterazioni sia minore del numero massimo
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            num_it = 0
            #Per ogni livello di temperatura faccio N iterazioni.
            while num_it < self.num_it:
                #Definisco la mossa che mi permette di generare una nuova soluzione
                #Per renderla quanto più casuale possibile ho deciso di non adottare un approccio k-opt fisso
                #ma di generare random il numero di elementi dell' array da scambiare
                candidate = list(self.cur_solution)
                l = random.randint(2, self.N - 1)
                i = random.randint(0, self.N - l)
                candidate[i : (i + l)] = reversed(candidate[i : (i + l)])
                self.accept(candidate)
                num_it += 1
            #Decremento il valore della temperatura e aumento il numero di iterazioni fatte
            self.T *= self.alpha
            self.iteration += 1
            #Aggiungo la soluzione corrente alla lista di soluzioni
            self.fitness_list.append(self.cur_fitness)

        print("Miglior doluzione ottenuta: ", self.best_fitness)
        
    def visualize_routes(self):
        """
        Rappresenta il percorso con la libreria maplotlib
        """
    
        visualize_tsp.plotTSP([self.best_solution], self.coords)

    def visualize_opt(self):
        berlin52 = [0,33,42,29,18,51,7,8,9,27,43,35,10,36,13,12,31,25,37,38,11,24,3,5,14,4,23,32,48,47,50,49,46,45,44,28,30,15,39,34,19,22,40,1,6,26,20,16,2,17,41,21]
        eil51 = [0,21,7,25,30,27,2,35,34,19,1,28,20,15,49,33,29,8,48,9,38,32,44,14,43,41,39,18,40,12,24,13,23,42,6,22,47,5,26,50,45,11,46,17,3,16,36,4,37,10,31]
        eil76 = [0,32,62,15,2,43,31,8,38,71,57,11,39,16,50,5,67,3,74,75,25,66,33,45,51,26,44,28,47,29,1,73,27,60,20,46,35,68,70,59,69,19,36,4,14,56,12,53,18,7,34,6,52,13,58,10,65,64,37,9,30,54,24,49,17,23,48,22,55,40,42,41,63,21,61,72]
        kroa100 = [0,46,92,27,66,57,60,50,86,24,80,68,63,39,53,1,43,49,72,67,84,81,94,12,75,32,36,4,51,77,95,38,29,47,99,40,70,13,2,42,45,28,33,82,54,6,8,56,19,11,26,85,34,61,59,76,22,97,90,44,31,10,14,16,58,73,20,71,9,83,35,98,37,23,17,78,52,87,15,93,21,69,65,25,64,3,96,55,79,30,88,41,7,91,74,18,89,48,5,62]
        val_opt = self.fitness(berlin52)
        print("Valore ottimo:", val_opt)

        visualize_tsp.plotTSP([berlin52], self.coords)
    
    
    def plot_learning(self):
        """
        Rappresenta su un grafico il valore della funzione obiettivo con l'aumentare del numero delle iterazioni
        """
        plt.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        plt.ylabel("Valore")
        plt.xlabel("Iterazioni")
        plt.show()
