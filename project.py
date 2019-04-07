from functools import reduce
from itertools import product

import numpy as np

import utils as ut
from operator import mul
inv_nucleotide = {v: k for k, v in ut.nucleotide.items()}
nucleobases = [k for k in inv_nucleotide]


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
    counts = {}
    for i in range(len(sequence)-k+1):
        try:
            counts[tuple(sequence[i:i + k])] += 1
        except Exception:
            counts[tuple(sequence[i:i + k])] = 1
    key = lambda item: item[0]
    sorted_dict = {k: v for k,v in sorted(counts.items(), key=key)}
    return sorted_dict


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
        dico[n_g] = reduce(lambda x, y: x * y, [frequences[i] for i in n_g],
                           number_n_grams)
    return dico


def simule_sequence(length, prop):
    """Generates a random sequence of length lg given the proportions of A,C,G and T.

    Parameters
    ----------
    prop : list of float
        proportions of A,C,G and T.
    length : int
        The size of a DNA sequence.
    Returns
    -------
    numpy.array
        A random sequence of length lg
    """
    liste = [np.full((round(length * p)), k) for k, p in enumerate(prop)]
    flattened = np.hstack(liste)
    return np.random.permutation(flattened)


def distance_counts(expected, observed):
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
    return np.linalg.norm([expected[k] - observed[k] for k in nucleotide])


def compare_simualtions(length, freqs, num_seq=1000):
    """
    compare the expected number of occurrences and the observed one by simulating a given number of sequences.

    Parameters
    ----------
    length : int
        The size of a DNA sequence.
    frequences : list of int
        The number of occurrences of the different elements of the sequence.
    num_seq : int
        the number of sequences to simulate.

    Returns
    -------

    """

    # obs_count = np.array([])
    pass


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
    matrix = np.array([[count_bigram(sequence, fst, snd) for snd in nucleobases]
                       for fst in nucleobases])
    return matrix / matrix.sum(axis=1).reshape(-1, 1)


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
    matrix = transition_matrix(sequence)
    seq = []
    proba = prop
    for _ in range(length):
        nb = np.random.choice(nucleobases, p=proba)
        proba = matrix[nb]
        seq.append(nb)
    return seq


def markov_proba(sequence, matrix, pi_k):
    """

    Parameters
    ----------

    Returns
    -------

    """
    def tmp(p, n):
        return p[0] * matrix[p[1]][n], n
    first_nb = sequence[0]
    initial_proba = pi_k[first_nb]
    return reduce(tmp, sequence[1:], (initial_proba, first_nb))[0]


def markov_proba_v2(word, position, probas, matrix):
    p = np.dot(probas,np.linalg.matrix_power(matrix,position))[word[0]]
    return reduce(mul, matrix[word[0]], 1) * p


def comptage_attendu_markov(k, length, frequences):
    """

    Parameters
    ----------

    Returns
    -------

    """
    pass
