from anneal import SimulatedAnnealing
import matplotlib.pyplot as plt
import random

#Funzione per leggere i nodi del file di testo in ingrsso
def read_coords(path):
    coords = []
    with open(path, "r") as f:
        for line in f.readlines():
            line = [float(x.replace("\n", "")) for x in line.split(" ")]
            coords.append(line)
    return coords


if __name__ == "__main__":
    #Chiamo la funzione di lettura dei nodi 
    coords = read_coords("a280.txt") 
    #Definisco i parametri dell' algoritmo da applicare all' istanza 
    sa = SimulatedAnnealing(coords, stopping_iter=50000,T= 10,alpha= 0.9,num_it= 100)
    #Applico l'algoritmo di simulated annealing
    sa.anneal()
    #Chiamo la funzione che mi traccia il grafico 
    sa.visualize_routes()
    #sa.visualize_opt()
    #Raffiguro i valori della funzione obiettivo in base al numero di iterazioni
    sa.plot_learning()
    
