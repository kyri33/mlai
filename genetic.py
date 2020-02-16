import random

POPULATION_SIZE = 100

GENES = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'

TARGET = 'I love GeeksforGeeks'

class Individual():

    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calc_fitness()

    @classmethod
    def mutated_gene(self):
        # Pick random gene
        global GENES
        gene = random.choice(GENES)
        return gene

    @classmethod
    def create_gnome(self):
        global TARGET
        chromosome = [self.mutated_gene() for _ in range(len(TARGET))]
        return chromosome

    def mate(self, par2):
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):
            prob = random.random()
            
            if prob < 0.45:
                child_chromosome.append(gp1)
            elif prob < 0.90:
                child_chromosome.append(gp2)
            else:
                child_chromosome.append(self.mutated_gene())
        
        return Individual(child_chromosome)
    
    def calc_fitness(self):
        global TARGET
        fitness = 0
        for gp1, gp2 in zip(self.chromosome, TARGET):
            if gp1 != gp2: fitness += 1
        
        return fitness

generation = 1

found = False
population = []

for _ in range(POPULATION_SIZE):
    gnome = Individual.create_gnome()
    population.append(Individual(gnome))

while not found:

    population = sorted(population, key=lambda x:x.fitness)
    if population[0].fitness <= 0:
        found = True
        break

    new_generation = []

    s = int((10 * POPULATION_SIZE) / 100)
    new_generation.extend(population[:s])

    s = int((90*POPULATION_SIZE) / 100)
    ind = int(POPULATION_SIZE * 0.5)
    for _ in range(s):
        par1 = random.choice(population[:ind])
        par2 = random.choice(population[:ind])
        child = par1.mate(par2)
        new_generation.append(child)

    population = new_generation

    print("Generation: {}\tString: {}\tFitness: {}". 
              format(generation, 
              "".join(population[0].chromosome), 
              population[0].fitness)) 
  
    generation += 1

print("Generation: {}\tString: {}\tFitness: {}". 
          format(generation, 
          "".join(population[0].chromosome), 
          population[0].fitness)) 