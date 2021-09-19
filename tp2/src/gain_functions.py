import math


def shannon(data):
    """
    H(S) = -(p_+) * log2(p_+) - (p_-) *log2(p_-)
    given p_+ as the probability of a given event as positive
    (relative frequency of positive events)
    and p_- analogous for negative events
    if p_+ = 0 => log2(p_+)=0 same for p_-

    Generalized for n values (not just positive and negative
    f(x) = -x * log2(x)
    H(S) = sum(f(p_v)) with p_v being the probability for each value v the attribute can take

    This function receives a dataFrame with one column and calculates its Shannon Entropy
    For example, if df is a training set, this function should be called as shannon(df[attribute])
    """

    def f(x):
        return -x * math.log2(x) if x != 0 else 0

    # List of all the relative probabilities for a certain attribute
    probabilies_list = __relative_frequencies(data).values()

    return sum(f(x) for x in probabilies_list)


def gini(data):
    return 1 - sum(__relative_frequencies(data).values())


def __relative_frequencies(df):
    """
    Input: DataFrame with just one column
    Generates a dictionary where the keys are each possible values
    and the value is the relative frequency of that key in the dataFrame
    """
    relative_frequencies = {}
    for k, v in df.value_counts().items():
        relative_frequencies[k] = v / len(df)
    return relative_frequencies
