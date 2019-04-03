from functools import reduce

import numpy as np

import utils as ut

inv_nucleotide = {v: k for k, v in ut.nucleotide.items()}


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


def inv_code(index, k):
    """Determines a DNA sequence based on the alphabetical order of its
    nucleotides.

    Parameters
    ----------
    index : int
        The alphabetical index of a DNA sequence.
    k : int
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
        expand, [4**i for i in range(k - 1, -1, -1)], ('', index))
    return int_to_str(sequence)


def n_grams(sequence, k):
    """Counts the number of occurrences of `k`-grams in a sequence.

    Parameters
    ----------
    sequence : str
        The DNA sequence to be analised.
    k : int
        The size of the n-grams to be considered.

    Returns
    -------
    dict of str: int
        The number of occurrences of the different `k`-grams in the sequence.

    """
    counts = {}
    for i in range(len(sequence) - k + 1):
        try:
            counts[sequence[i:i + k]] += 1
        except Exception:
            counts[sequence[i:i + k]] = 1
    return counts
