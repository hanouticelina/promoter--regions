from functools import reduce
from itertools import product
import math
from scipy.stats import poisson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils as ut
from operator import mul
inv_nucleotide = {v: k for k, v in ut.nucleotide.items()}
nucleobases = [k for k in inv_nucleotide]

d={'k':1,'f':2, 'j':4,'g':9}
len(d)
def logproba(sequence, probas):
    """Returns the log-probability of an integer sequence.

    Parameters
    ----------
    sequence : list of int
        A number sequence.

    probas : tuple of float
        The probabilities of the unique sequence values.

    Returns
    -------
    float
        The log-probability that the sequence is obtained.

    """
    return sum([np.log(probas[nb]) for nb in sequence])


def logprobafast(frequencies):
    """Returns the (improved) log-probability of an integer sequence
    from the frequency of its elements.

    Parameters
    ----------
    frequencies : list of int
        The number of occurrences of the different elements of the sequence.

    Returns
    -------
    float
        The log-probability that the sequence is obtained when estimating
        its elements probability with their frequencies.

    """
    return (np.array(frequencies) * np.log(frequencies)).sum() - sum(frequencies) * np.log(sum(frequencies))


def code(word, size):
    """Determines the index of a DNA sequence based on its alphabetical order.

    Parameters
    ----------
    word : str of letters
        The character sequence.

    size : int
        The number of characters in the word.

    Returns
    -------
    int
        The alphabetical index of the word based on its nucleobase components.

    See Also
    --------
    nucleotides : the map with the value for each nucleobase.

    """
    return reduce(lambda p, n: 4 * p + n, [ut.nucleotide[c] for c in word])


def str_to_int(string):
    """Converts a DNA sequence to a list of integers.

    Parameters
    ----------
    string : str
        The string of nucleobases.

    Returns
    -------
    list of int
        A list with the integer value for each nucleobase in the string.

    """
    return [ut.nucleotide[nb] for nb in string]


def int_to_str(sequence):
    """Converts a numerical string to a DNA sequence.

    Parameters
    ----------
    sequence : str
        The string with numerical characters.

    Returns
    -------
    str
        The DNA sequence corresponding to the numerical string.

    """
    return ''.join([inv_nucleotide[int(n)] for n in sequence])


def inv_code(index, length):
    """Determines a DNA sequence based on the alphabetical order of its
    nucleotides.

    Parameters
    ----------
    index : int
        The alphabetical index of a DNA sequence.
    length : int
        The size of a DNA sequence.

    Returns
    -------
    str
        The `k`-size DNA sequence corresponding to the given alphabetical
        index.

    """
    def expand(p, n):
        """Adds the integer value of a nucleobase to a DNA sequence portion
        already converted.

        Parameters
        ----------
        p : (str, int)
            The pair containing the beginning portion of DNA sequence obtained
            so far, and the remaining of the index to be converted.
        n : int
            The power of 4 associated to the next nucleobase.

        Returns
        -------
        str
            The numerical sequence corresponding to the enlarged beginning of
            the DNA sequence.
        int
            The remaining index to be converted.

        """
        return p[0] + str(p[1] // n), p[1] % n

    sequence, _ = reduce(
        expand, [4**i for i in range(length - 1, -1, -1)], ('', index))
    return int_to_str(sequence)


def k_grams_occurrences(sequence, k):
    """Counts the number of occurrences of `k`-grams in a sequence.

    Parameters
    ----------
    sequence : str
        The DNA sequence to be analysed.
    k : int
        The size of the k-grams to be considered.

    Returns
    -------
    dict of str: int
        The number of occurrences of the different `k`-grams in the sequence.

    """
    k_grams = all_k_grams(k)
    counts = {tuple(k) : 0 for k in k_grams}
    for i in range(len(sequence)-k+1):
        try:
            counts[tuple(sequence[i:i + k])] += 1
        except Exception:
            pass
    return counts


def all_k_grams(k, alph=nucleobases):
    """Returns all k-grams given the alphabet

    Parameters
    ----------
    k : int
        The size of the k-grams to be considered.
    alph : str
        The alphabet.

    Returns
    -------
    list of str
        all possible k-grams given the alphabet.

    """
    return list(product(alph, repeat=k))

def nucleotides_proba(sequence, frequencies):
    return reduce(lambda x, y: x * y, [frequencies[nb] for nb in sequence])

def comptage_attendu(k, length, frequences):
    """Returns the expected number of occurrences of the different k-grams given the frequencies of letters in the DNA sequence

    Parameters
    ----------
    k : int
        The size of the k-grams to be considered.
    length : int
        The size of a DNA sequence.
    frequences: list of int
        The number of occurrences of the different elements of the sequence.
    Returns
    -------
    dict of str: int
        the expected number of occurrences of the different k-grams.

    """
    n_grams = all_k_grams(k)
    number_n_grams = length - k + 1
    dico = {}
    for n_g in n_grams:
        dico[n_g] = nucleotides_proba(n_g, frequences) * number_n_grams
    return dico


def simule_sequence(length, props):
    """Generates a random sequence of length lg given the proportions of A,C,G and T.

    Parameters
    ----------
    props : list of float
        proportions of A,C,G and T.
    length : int
        The size of a DNA sequence.
    Returns
    -------
    numpy.array
        A random sequence of length lg
    """
    liste = [np.full((int(round(length * p))), k) for k, p in enumerate(props)]
    flattened = np.hstack(liste)
    return np.random.permutation(flattened)


def compare_simulations(length, freqs, ks, num_seq=1000):
    """
    compare the expected number of occurrences and the observed one by simulating a given number of sequences.

    Parameters
    ----------
    length : int
        The size of a DNA sequence.
    frequences : list of int
        The number of occurrences of the different elements of the sequence.
    ks : list of int
        The lengths of the n-grams to be considered.
    num_seq : int
        the number of sequences to simulate.

    Returns
    -------

    """

    observed = {k: None for k in ks}
    expected = {k: None for k in ks}
    for k in ks:
        exp = np.zeros((num_seq,len(comptage_attendu(k,length,freqs))))
        obs = np.zeros_like(exp)
        for i in range(num_seq):
            sequence = simule_sequence(length, freqs)
            obs[i] = list(k_grams_occurrences(sequence,k).values())
            exp[i] = list(comptage_attendu(k,len(sequence),freqs).values())
        observed[k] = list(np.mean(obs, axis = 0))
        expected[k] = list(np.mean(exp, axis = 0))
    return observed, expected

def p_empirique(length, n_gram, freqs, nb_simulation=1000):
    """
    Parameters
    ----------

    Returns
    -------

    """

    nb_observed ={}
    word = tuple(str_to_int(n_gram))
    for _ in range(nb_simulation):
        dict = k_grams_occurrences(simule_sequence(length, freqs), len(word))
        nb = 0
        if(word in dict.keys()):
             nb = dict[word]
        if nb in nb_observed:
            nb_observed[nb] +=1
        else:
            nb_observed[nb] = 1
    sorted_obs = {k: v for k, v in sorted(nb_observed.items())}

    keys = list(sorted_obs.keys())
    max_occu = keys[-1] + 1
    values = np.zeros(max_occu + 1)
    values[keys] = list(sorted_obs.values())
    values = np.cumsum(values[::-1])[::-1] / nb_simulation
    return values

def plot_histogram(sequence, freqs, nb_simulation=1000, expected=False):
    """

    Parameters
    ----------

    Returns
    -------

    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    p_emp = {}
    p_dint = {}
    p_nt = {}
    words = ["ATCTGC", "ATATAT", "AAAAAA", "TTTAAA"]
    positions = [(0,0), (0,1), (1,0), (1,1)]

    # Distribution stationnaire de la chaîne de Markov associée
    pi_k = stationary_distribution(freqs, transition_matrix(sequence), 0.00001, verbose=False)

    for word in words:
        p_emp[word] = p_empirique(len(sequence), word, freqs, nb_simulation)
        if expected is True:
            p_dint[word] = dinucleotides_proba(str_to_int(word), transition_matrix(sequence), pi_k)
            p_nt[word] = nucleotides_proba(str_to_int(word), freqs)

    for pos, word in zip(positions, p_emp.keys()):
        ks = np.arange(len(p_emp[word]))
        axes[pos].grid(True)
        axes[pos].set_title("Distribution des occurrences de " + word)
        axes[pos].set_xlabel("Occurrences du mot")
        axes[pos].set_ylabel("Probabilité empirique estimée")
        axes[pos].bar(ks, p_emp[word])

        if expected is True:
            # Paramètre de la loi de Poisson
            mu_dint = p_dint[word] * (len(sequence) - len(word) + 1)
            mu_nt = p_nt[word] * (len(sequence) - len(word) + 1)
            axes[pos].scatter(ks, geq_poisson_probability(ks, mu_dint), zorder=2)
            axes[pos].scatter(ks, geq_poisson_probability(ks, mu_nt), zorder=3)
            axes[pos].legend(['Loi de Poisson (dinucléotides)', \
            'Loi de Poisson (nucléotides)', 'Distribution empirique'])

        extent = axes[pos].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig("plots/histogram_" + word + ".png", bbox_inches=extent.expanded(1.1, 1.2))

def plot_scatter(files, ks):
    chromos_list = []
    fig, axes = plt.subplots(len(ks), len(files), figsize=(15, 15))
    plt.subplots_adjust(hspace=0.45, wspace=0.35)
    for i in range(len(files)):
        _, chromos, _, freqs = ut.read_file(files[i])
        chromos_list += chromos
        for ind_k, k in enumerate(ks):
            obs = k_grams_occurrences(chromos, k)
            exp = comptage_attendu(k, len(chromos), freqs)
            observed, expected = ut.encode_file(obs, exp)
            ut.plot(xs=observed, ys=expected, xlabel="Nombre observé", \
                    ylabel="Nombre attendu", \
                    title="Fichier: " + files[i][10:] + ", k = " + str(k), \
                    ax=axes[ind_k, i])
    return chromos_list

def distance_counts(xs, ys):
    """Compute the distance between the expected number of occurrences and the observed one.

    Parameters
    ----------
    expected : numpy.array
        the expected number of occurrences of the different elements of the sequence.
    observed : numpy.array
        the expected number of occurrences of the different elements of the sequence.
    Returns
    -------
    float
        the norm of the array (expected - observed).

    """
    dists = []
    for x, y in zip(xs, ys):
        dists.append(x - y)
    return np.linalg.norm(dists), np.std(np.abs(dists), ddof=1)


def plot_counts(sequence, freqs):
    """

    Parameters
    ----------

    Returns
    -------

    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    ks = [2, 4, 6, 8]
    positions = [(0,0), (0,1), (1,0), (1,1)]

    length = len(sequence)
    # Matrice de transition de la chaîne de Markov associée
    tmatrix = transition_matrix(sequence)
    # Distribution stationnaire de la chaîne de Markov associée
    pi_k = stationary_distribution(freqs, tmatrix, 0.00001, verbose=False)

    print("Méthode des moindres carrés des distances à l'observation")
    print("=========================================================")

    for pos, k in zip(positions, ks):
        # axes[pos].grid(True)
        axes[pos].set_title("Longueur des mots : k = " + str(k))
        axes[pos].set_xlabel("Indice lexicographique du mot")
        axes[pos].set_ylabel("Comptage attendu")

        nt_counts = comptage_attendu(k, length, freqs)
        dint_counts = comptage_attendu_markov(k, length, tmatrix, pi_k)
        obs_counts = k_grams_occurrences(sequence, k)

        nt, dint = ut.encode_file(nt_counts, dint_counts)
        obs = list(obs_counts.values())
        xs = np.arange(len(nt_counts))

        mse_nt, std_nt = distance_counts(obs, nt)
        mse_dint, std_dint = distance_counts(obs, dint)

        print("\nPour k =", k, ":")
        print("============")
        print("Modèle\t\t|\tSomme des carrés\t|\tEcart type")
        print("------------------------------------------------------------------------")
        print("Nucléotides\t|\t", round(mse_nt, 4), "\t\t|\t", round(std_nt, 4))
        print("Dinucléotides\t|\t", round(mse_dint, 4), "\t\t|\t", round(std_dint, 4))

        # Paramètre de la loi de Poisson
        axes[pos].scatter(xs, obs, zorder=1)
        axes[pos].scatter(xs, nt, zorder=2)
        axes[pos].scatter(xs, dint, zorder=3)
        axes[pos].legend(['Observations', 'Modèle de nucléotides', \
        'Modèle de dinucléotides'])

def count_bigram(sequence, first, second):
    """
    counts the number of occurrences of a given bigram  in a sequence.

    Parameters
    ----------
    sequence : str
        The DNA sequence to be analysed.
    first : char
        the first letter of the bigram to be considered.
    second : char
        the second letter of the bigram to be considered.
    Returns
    -------
    int
        the number of occurrences of the bigram.

    """
    count = 0
    cmp = False
    for char in sequence:
        if cmp is True:
            cmp = False
            if char == second:
                count += 1
        if char == first:
            cmp = True
    return count


def transition_matrix(sequence):
    """
    Compute the transition matrix of the markov chain model of the frequencies of overlapping dinucleotides.

    Parameters
    ----------
    sequence : str
        The DNA sequence to be analysed.

    Returns
    -------
    numpy.ndarray
        the transition matrix.

    """
    tmatrix = np.array([[count_bigram(sequence, fst, snd) for snd in nucleobases]
                       for fst in nucleobases])
    return tmatrix / tmatrix.sum(axis=1).reshape(-1, 1)


def simule_sequence_markov(length, prop, sequence):
    """
    simulates a sequence with the dinucleotide model.

    Parameters
    ----------
    sequence : str
        The DNA sequence to be analysed.
    length : int
        the length of the sequence to be simulated.
    prop : list of float
        the proportions of the different elements of the sequence.
    Returns
    -------
        the simulated sequence.
    """
    tmatrix = transition_matrix(sequence)
    seq = []
    proba = prop
    for _ in range(length):
        nb = np.random.choice(nucleobases, p=proba)
        proba = tmatrix[nb]
        seq.append(nb)
    return seq


def dinucleotides_proba(sequence, tmatrix, pi_k):
    """

    Parameters
    ----------

    Returns
    -------

    """
    def tmp(p, n):
        return p[0] * tmatrix[p[1]][n], n
    first_nb = sequence[0]
    initial_proba = pi_k[first_nb]
    return reduce(tmp, sequence[1:], (initial_proba, first_nb))[0]


def comptage_attendu_markov(k, length, tmatrix, pi):
    """

    Parameters
    ----------
    pi: tuple of int
        The distribution to use for the first nucleobase in the `k`-gram.

    Returns
    -------

    See Also
    --------
    stationary_distribution

    """
    n_grams = all_k_grams(k)
    number_n_grams = length - k + 1
    dico = {}
    for n_g in n_grams:
        dico[n_g] = number_n_grams * dinucleotides_proba(n_g, tmatrix, pi)
    return dico

def stationary_distribution(pi_0, tmatrix, epsilon, verbose=True):
    """
    """
    pi_k = pi_0
    k = 0
    while True:
        k += 1
        pi_kp1 = np.dot(pi_k, tmatrix)
        if np.abs(pi_kp1 - pi_k).sum() < epsilon:
            if verbose is True:
                print("Convergence after", k, "iterations")
            return pi_kp1
        pi_k = pi_kp1

def p_empirique_dinucleotides(length, n, counts, k, word, freqs, nb_simulation=1000):
    """
    Parameters
    ----------

    Returns
    -------

    """
    nb_observed = 0
    for _ in range(nb_simulation):
        dict = k_grams_occurrences(simule_sequence(length,freqs),len(word))
        if(word in dict.keys() and dict[word] >= n ): nb_observed +=1
    return nb_observed

def geq_poisson_probability(n, mu):
    return 1 - poisson.cdf(n, mu=mu)


def gap_probabilities(length, k, counts, probas):
    nb_pos = length - k + 1
    return [geq_poisson_probability(n, nb_pos*p) for n, p in zip(counts, probas)]

def unexpected_words(sequence,freqs,k,seuil):
    occ = k_grams_occurrences(sequence,k)
    nb_pos = len(sequence)- k + 1
    pik = stationary_distribution(freqs, transition_matrix(sequence), 0.00001)
    words = []
    for w in occ.keys():
        p = dinucleotides_proba(w,transition_matrix(sequence),pik)
        proba = geq_poisson_probability(occ[w],p*nb_pos)
        if proba < seuil:
            words.append(int_to_str(w))
            print("word: "+int_to_str(w)+"\t\tOccurrences: "+str(occ[w])+"\t\tP(N >= {:d}) = {:.4f}".format(occ[w], proba), sep="")
