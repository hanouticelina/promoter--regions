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

def plot(xs, ys, xlabel, ylabel, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    min_v = min(min(ys), min(xs))
    max_v = max(max(ys), max(xs))
    ax.scatter(ys, xs, c="blue", marker="o")
    ax.plot([min_v, max_v], [min_v, max_v], color="red", zorder=2)
    ax.set_title(title)

def encode_file(obs, exp):
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
