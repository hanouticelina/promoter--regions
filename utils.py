import io
import math
import operator
from itertools import product
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np
nucleotide = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
nucleotide_indetermine = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1}


import project




def decode_sequence(sequence):
    inv_nucleotide = {v: k for k, v in nucleotide_indetermine.items()}
    to_str = ""
    for i in sequence:
        if(i in inv_nucleotide):
            to_str += inv_nucleotide[i]
        else:
            to_str += 'N'
    return to_str


def encode_sequence(string):
    to_list = []
    for base in string:
        if(base in nucleotide_indetermine):
            to_list.append(nucleotide_indetermine[base])
    return to_list


def read_fasta(fasta_filepath):
    fasta_file = io.open(fasta_filepath, 'r')
    current_sequence = ""
    sequences_dict = {}
    for line in fasta_file.readlines():
        if(line[0] == '>'):
            current_sequence = line
            sequences_dict[line] = []
        else:
            for nucl in line:
                if(nucl in nucleotide_indetermine):
                    sequences_dict[current_sequence].append(
                        nucleotide_indetermine[nucl])

    return sequences_dict


def nucleotide_count(sequence):
    count = [0 for k in nucleotide]
    for nucl in sequence:
        if(nucl >= 0):
            count[nucl] += 1
    return count


def nucleotide_frequency(sequence):
    count = [0 for k in nucleotide]
    n_nucl = 0.
    for nucl in sequence:
        if(nucl >= 0):
            count[nucl] += 1
            n_nucl += 1.
    return count / (np.sum(count))

def plot(file,k):
    f = read_fasta(file)
    _,chromos_flattened, _ , probas = read_file(file)
    observed = project.k_grams_occurrences(chromos_flattened,k)
    expected = project.comptage_attendu(k,len(chromos_flattened),probas)
    expected_values = [expected[key]  for key in expected.keys() if key in observed.keys()]
    observed_values = [observed[key]  for key in expected.keys() if key in observed.keys()]
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.grid(True)
    plt.xlabel("Nombre attendu")
    plt.ylabel("Nombre observe")
    ax.scatter(expected_values, observed_values, c="blue", marker="o")
    x = np.linspace(np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()]))
    ax.plot(x,x, color="red")
    ax.set_title("Graphique 2D representant l'ecart entre le comptage attendu et le comtage observé sur le fichier: "+file[10:])
    fig.savefig("plots/"+file[10:]+"_"+str(k)+".png")

def read_file(file):
    cerevisae = read_fasta(file)
    chromos = [seq for seq in cerevisae.values()]
    chromos_flattened = [el for sub in chromos for el in sub]
    freqs = nucleotide_count(chromos_flattened)
    probas = nucleotide_frequency(chromos_flattened)
    return cerevisae,chromos_flattened,freqs,probas
