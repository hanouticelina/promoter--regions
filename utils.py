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

def plot(file,k,observed,expected):
    min_v = float("inf")
    max_v = -float("inf")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_axisbelow(True)
    ax.grid(True)
    plt.xlabel("Nombre attendu")
    plt.ylabel("Nombre observe")
    min_v = min(min(expected), min(observed), min_v)
    max_v = max(max(expected), max(observed), max_v)
    ax.scatter(expected, observed, c="blue", marker="o")
    ax.plot([min_v, max_v], [min_v, max_v], color="red", zorder=2)
    ax.set_title("Graphique 2D representant l'ecart entre le comptage attendu et le comtage observé sur le fichier: "+file[10:]+"et pour k = "+str(k))

def encode_file(sequence, k, probas, obs, exp):
    observed = [obs[k] for k in obs.keys()]
    expected = [exp[k] for k in exp.keys()]
    return observed, expected

def read_file(file):
    genome = read_fasta(file)
    chromos = [seq for seq in genome.values()]
    chromos_flattened = [el for sub in chromos for el in sub]
    counts = nucleotide_count(chromos_flattened)
    freqs = nucleotide_frequency(chromos_flattened)
    return genome, chromos_flattened, counts, freqs
