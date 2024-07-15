import collections
import itertools
import math
from typing import List, Optional, Literal, Dict
import logging

from arc.types.base_types import Register, RegisterType
from arc.types.syllable import Syllable
from arc.types.word import WordType, Word
from arc.types.lexicon import LexiconType
from arc.types.stream import StreamType, Stream

from arc.controls.common import get_oscillation_patterns

from arc.core.lexicon import make_lexicon_generator

from arc.controls.common import *


logger = logging.getLogger(__name__)


def shuffled_struct_stream(n_words=4, n_sylls_per_word=3, n_repetitions=4):
    # TODO: make faster and easier to read
    n_sylls_total = n_sylls_per_word * n_words  # number of syllables in a lexicon
    n_iters = n_sylls_total * n_repetitions  # number of repetitions in a trial
    syll_id = np.arange(n_sylls_total).reshape((n_words, int(n_sylls_total / n_words)))
    while True:
        stream = np.repeat(syll_id, n_iters, axis=0)
        np.random.shuffle(stream)
        v = list(stream)
        for n in range(1, len(v)-1):
            if v[n][0] == v[n-1][0]:
                k = v[n:]
                l = [i for i, el in enumerate(k) if el[0] != v[n][0]]
                if not l:
                    continue
                else:
                    l = l[0]
                v[n] = v[n+l]
                v[n+l] = v[n-1]
        x = [i[0] for i in v]
        check = [sum(1 for _ in group) for _, group in itertools.groupby(x)]
        if all(i == 1 for i in check):
            break
    s = [j for k in v for j in k]
    return s


def shuffled_random_stream(n_words=4, n_sylls_per_word=3, n_repetitions=4):
    # TODO: make faster and easier to read
    n_sylls_total = n_sylls_per_word * n_words  # number of syllables in a lexicon
    n_iters = n_sylls_total * n_repetitions  # number of repetitions in a trial:
    syll_id = np.arange(n_sylls_total)
    while True:
        stream = np.repeat(syll_id, n_iters)
        np.random.shuffle(stream)
        v = list(stream)
        for n in range(1, len(v)-1):
            if v[n] == v[n-1]:
                v[n] = v[n+1]
                v[n+1] = v[n-1]
        check = [sum(1 for _ in group) for _, group in itertools.groupby(v)]
        if all(i == 1 for i in check):
            break
    return v


def transitional_p_matrix(v):
    # TODO: make (faster and) easier to read
    n = 1 + max(v)
    M = [[0] * n for _ in range(n)]
    for (i, j) in zip(v, v[1:]):
        M[i][j] += 1
    for r in M:
        s = sum(r)
        if s > 0:
            r[:] = [i/s for i in r]
    return M


def pseudo_walk_tp_random(P, T, v, S, N, n_words=4, n_sylls_per_word=3):
    # TODO: make faster and easier to read
    t = []
    for iTrip in range(n_words):
        for iPoss in range(n_sylls_per_word):
            if not v and not t:
                if N:
                    x = [N]
                    N = []
                else:
                    x = random.sample(P[iPoss], 1)
                t += x
            else:
                iLoop = 0
                while iLoop < 1000:
                    x = random.sample([i for i in P[iPoss] if i not in t], 1)[0]
                    if not t:
                        s = v[-1]
                    else:
                        s = t[-1]
                    if [s, x] not in S and [s, x] in T[iPoss-1]:
                        S.append([s, x])
                        t.append(x)
                        break
                    else:
                        iLoop += 1
    if iLoop < 1000:
        v += t
    return v, S, t


def pseudo_rand_tp_random(n_words=4, n_sylls_per_word=3, n_repetitions=4):
    # TODO: make faster and easier to read
    n_sylls_total = n_sylls_per_word * n_words  # number of syllables in a lexicon
    n_iters = n_repetitions  # number of repetitions in a trial
    n_loop = 1000  # ???
    while True:
        p1 = np.repeat(range(n_sylls_total), math.ceil(n_iters / int(n_sylls_total / n_words))).tolist()
        p2 = np.repeat(range(n_sylls_total), math.ceil(n_iters / int(n_sylls_total / n_words))).tolist()
        p3 = np.repeat(range(n_sylls_total), math.ceil(n_iters / int(n_sylls_total / n_words))).tolist()
        M = np.zeros((n_sylls_total, n_sylls_total))
        mask = (np.ones((n_sylls_total, n_sylls_total)) - np.diag(np.ones(n_sylls_total))).astype(bool)
        v = []
        for _ in range(n_iters):
            if not v:
                p = random.sample(range(n_sylls_total), n_sylls_total)
                m = np.array(transitional_p_matrix(p))
            else:
                i_loop = 0
                while i_loop < n_loop:
                    try:
                        s = v[-1]
                        p = []
                        m = np.zeros((n_sylls_total, n_sylls_total))
                        N = np.copy(M)
                        for idx in range(n_sylls_total):
                            t = list(np.where(N[s] == N[s].min())[0])
                            t.remove(s)
                            if idx in list(range(n_sylls_total)[0::3]):
                                t = [i for i in p1 if i in t and i not in p]
                                if not t and i_loop > n_loop/100:
                                    t = list(np.where(N[s] == np.sort(N[s])[1])[0])
                                    t = [i for i in p1 if i in t and i not in p]
                                if not t and i_loop > n_loop/10:
                                    t = [i for i in p1 if i != s]                        
                                    t = [i for i in p1 if i in t and i not in p]
                                x = np.array(list(map(list(t).count, set(t))))
                                n = [list(set(t))[i] for i in list(np.where(x == x.max())[0])]
                                p += random.sample(n, 1)
                            elif idx in list(range(n_sylls_total)[1::3]):
                                t = [i for i in p2 if i in t and i not in p]
                                if not t and i_loop > n_loop/100:
                                    t = list(np.where(N[s] == np.sort(N[s])[1])[0])
                                    t = [i for i in p2 if i in t and i not in p]
                                if not t and i_loop > n_loop/10:
                                    t = [i for i in p2 if i != s]
                                    t = [i for i in p2 if i in t and i not in p]
                                x = np.array(list(map(list(t).count, set(t))))
                                n = [list(set(t))[i] for i in list(np.where(x == x.max())[0])]
                                p += random.sample(n, 1)
                            elif idx in list(range(n_sylls_total)[2::3]):
                                t = [i for i in p3 if i in t and i not in p]
                                if not t and i_loop > n_loop/100:
                                    t = list(np.where(N[s] == np.sort(N[s])[1])[0])
                                    t = [i for i in p3 if i in t and i not in p]
                                if not t and i_loop > n_loop/10:
                                    t = [i for i in p3 if i != s]
                                    t = [i for i in p3 if i in t and i not in p]
                                x = np.array(list(map(list(t).count, set(t))))
                                n = [list(set(t))[i] for i in list(np.where(x == x.max())[0])]
                                p += random.sample(n, 1)    
                            m[s][p[idx]] = m[s][p[idx]] + 1
                            s = p[-1]
                            N += m
                        break
                    except:
                        i_loop += 1
            if len(p) == n_sylls_total:
                M += m
                v += p
            else:
                break
            if all(i in p1 
                   for i in p[0::3]) and all(i in p2 
                                             for i in p[1::3]) and all(i in p3 
                                                                       for i in p[2::3]):
                for i in range(n_words):
                    p1.remove(p[0::3][i])
                    p2.remove(p[1::3][i])
                    p3.remove(p[2::3][i])
            else:
                break
        if all(np.diagonal(M) == np.zeros(n_sylls_total)) and len(v) == n_sylls_total * n_iters:
            if M[mask].min() == math.floor(n_iters * n_sylls_total / len(M[mask])):
                if M[mask].max() == math.ceil(n_iters * n_sylls_total / len(M[mask])):
                    M = np.array(transitional_p_matrix(v))
                    break
    return v, M


def pseudo_rand_tp_random_position_controlled(n_words=4, n_sylls_per_word=3, n_repetitions=4):
    # TODO: make faster and easier to read
    n_sylls_total = n_sylls_per_word * n_words  # number of syllables in a lexicon
    n_iters = n_repetitions  # number of repetitions in a trial
    P = [list(range(i, n_sylls_total, n_sylls_per_word)) for i in range(n_sylls_per_word)]
    I = list(range(n_sylls_per_word))
    B = I[:]
    B.append(B.pop(0))
    T = [[list(i) for i in list(itertools.product(*[P[j], P[k]]))] for j, k in zip(I, B)]
    while True:
        V = []
        while len(V) < n_sylls_total * n_iters:
            v = []
            t = []
            if not V:
                N = []
                S = []
            else:
                N = [i for i in T[2] if i not in S][0][1]
                S = []
            while len(v) < len(T) * len(T[0]):
                v, S, t = pseudo_walk_tp_random(P, T, v, S, N)
                if len(t) < n_sylls_total:
                    v = []
                    S = []
                    t = []
            V += v
        V.append(V[0])
        M = np.array(transitional_p_matrix(V))
        V.pop()
        if all(set(i) <= set([1/n_words, 0]) for i in M):
            break
    return V, M


def pseudo_rand_tp_struct(n_words=4, n_sylls_per_word=3, n_repetitions=4):
    # TODO: make faster and easier to read
    n_sylls_total = n_sylls_per_word * n_words  # number of syllables in a lexicon
    n_iters = n_repetitions  # number of repetitions in a trial
    while True:
        P = list(itertools.permutations(np.arange(n_words)))
        S = [[*(i for _ in range(math.ceil(n_iters / len(P))))] for i in P]
        M = np.zeros((n_words, n_words))
        v = []
        c = 2
        mask = (np.ones((n_words, n_words)) - np.diag(np.ones(n_words))).astype(bool)
        for _ in range(n_iters):
            R = [S[i] for i in range(len(S)) if len(S[i]) == max(map(len, S))]
            if not v:
                p = list(list(random.sample(R, 1)[0])[0])
                m = np.array(transitional_p_matrix(p))
            else:
                s = v[-1]
                n = list(np.where(M[s] == M[s].min())[0])
                n.remove(s)
                if not n:
                    n = list(np.where(M[s] == np.sort(M[s])[1])[0])
                T = [j for j in P if any(k for k in [(s, i) for i in n]) in bytes(j) and j[0] != s]
                t = [i[0] for i in [[i for i in T if i in j] for j in R] if i]
                if not t:
                    t = T
                while True:
                    if collections.Counter(M[mask])[c - 1] >= len(M[mask]) - n_words:
                        c += 1
                    if t:
                        p = list(random.sample(t, 1)[0])
                        m = np.array(transitional_p_matrix(p))
                        m[s][p[0]] = m[s][p[0]] + 1
                        if c not in M + m:
                            break
                        else:
                            t.remove(tuple(p))
                    else:
                        c += 1
                        T = [i for i in P if any(j for j in [(s, k) for k in n]) in bytes(i) and i[0] != s]
                        t = [i[0] for i in [[i for i in T if i in j] for j in R] if i]
                        if not t:
                            t = T
            M += m
            v += p
            r = P.index(tuple(p))
            if tuple(p) in S[r]:
                S[r].remove(tuple(p))
            else:
                break
        if all(np.diagonal(M) == np.zeros(n_words)) and len(v) == n_words * n_iters:
            if M[mask].min() == math.floor(n_iters * n_words / len(M[mask])) - 1:
                if M[mask].max() == math.ceil(n_iters * n_words / len(M[mask])):
                    v.append(v[0])
                    M = np.array(transitional_p_matrix(v))
                    v.pop()
                    break
    return v, M


def sample_syllable_randomization(
        lexicon: LexiconType,
        n_repetitions: int = 4,
        max_tries=1000,
        tp_mode: Literal["word_structured", "position_controlled", "random"] = "word_structured") -> List[Syllable]:
    if tp_mode == "word_structured":
        elements = [word for word in lexicon]
        rand_func = pseudo_rand_tp_struct
    elif tp_mode == "position_controlled":
        elements = [syllable for word in lexicon for syllable in word]
        rand_func = pseudo_rand_tp_random_position_controlled
    elif tp_mode == "random":
        elements = [syllable for word in lexicon for syllable in word]
        rand_func = pseudo_rand_tp_random
    else:
        raise ValueError(f"tp_mode '{tp_mode}' unknown.")

    randomized_indexes_list = []
    n_syllables = len(lexicon[0].syllables)
    n_words = len(lexicon)

    for _ in range(max_tries):
        randomized_indexes, _ = rand_func(n_words=n_words, n_sylls_per_word=n_syllables, n_repetitions=n_repetitions)
        if randomized_indexes not in randomized_indexes_list:
            randomized_indexes_list.append(randomized_indexes)
            if tp_mode == "word_structured":
                yield [syllable for index in randomized_indexes for syllable in elements[index]]
            else:
                yield [elements[index] for index in randomized_indexes]


def compute_rhythmicity_index_sylls_stream(stream, patterns):
    count_patterns = []
    patterns = [tuple(pat) for pat in patterns]
    for feature_stream in list(zip(*[syllable.info["binary_features"] for syllable in stream])):
        c = 0
        for iSyll in range(len(feature_stream) - max(len(i) for i in patterns)):
            if any(i == feature_stream[iSyll: iSyll + len(i)] for i in patterns):
                c += 1
        count_patterns.append(c / (len(feature_stream) - max(len(i) for i in patterns)))

    return count_patterns


def make_stream_from_lexicon(lexicon: Register[str, Word],
                             max_rhythmicity: Optional[float] = None, max_tries_randomize=10, n_repetitions: int = 4,
                             tp_mode: Literal["word_structured", "position_controlled", "random"] = "word_structured"):

    for sylls_stream in sample_syllable_randomization(lexicon, max_tries=max_tries_randomize, tp_mode=tp_mode,
                                                      n_repetitions=n_repetitions):
        patterns = get_oscillation_patterns(len(lexicon[0].syllables))
        rhythmicity_indexes = compute_rhythmicity_index_sylls_stream(sylls_stream, patterns)

        if max_rhythmicity is None or (max(rhythmicity_indexes) <= max_rhythmicity):
            i_labels = enumerate(lexicon.info["syllables_info"]["syllable_feature_labels"])
            feature_labels = [f"phon_{i_phon+1}_{label}" for i_phon, labels in i_labels for label in labels]

            return Stream(
                id="".join([syll.id for syll in sylls_stream]),
                syllables=sylls_stream,
                info={
                    "rhythmicity_indexes": {k: float(v) for k, v in zip(feature_labels, rhythmicity_indexes)},
                    "lexicon": lexicon,
                    "stream_tp_mode": tp_mode,
                    **lexicon.info,
                }
            )


def make_stream_from_words(words: Register[str, Word],
                           n_words: int = 4,
                           max_word_overlap: int = 1,
                           max_lexicons: int = 10,
                           max_rhythmicity: float = 0.1,
                           max_tries_randomize: int = 10,
                           tp_mode: Literal["word_structured", "position_controlled", "random"] = "word_structured"
                           ) -> Optional[StreamType]:

    l_gen = make_lexicon_generator(words=words,
                                   n_words=n_words, max_overlap=max_word_overlap, max_yields=max_lexicons)

    for lex in l_gen:
        maybe_stream: Optional[StreamType] = make_stream_from_lexicon(lex, max_rhythmicity=max_rhythmicity,
                                                                      max_tries_randomize=max_tries_randomize,
                                                                      tp_mode=tp_mode)

        if maybe_stream:
            return maybe_stream


def get_stream_syllable_stats(stream: StreamType) -> Dict:
    d = {}
    for syllable in stream:
        if syllable.id in d:
            d[syllable.id] += 1
        else:
            d[syllable.id] = 1

    return d


def make_streams(
        lexicons: List[LexiconType],
        max_rhythmicity: Optional[float] = None,
        stream_length: int = 32,
        max_tries_randomize: int = 10,
        tp_modes: tuple = ("random", "word_structured", "position_controlled"),
        require_all_tp_modes: bool = True
) -> RegisterType:
    """_summary_

    Args:
        lexicons (List[LexiconType]): A list of lexicons used as a basis for generatng the streams
        max_rhythmicity (Optional[float], optional): check rhythmicity and discard all streams that have at least one feature with higher PRI than this number. Defaults to None.
        stream_length (int, optional): how many syllables are in a stream in multiples of syllables in the lexicon. Defaults to 4.
        max_tries_randomize (int, optional): if max_rhythmicity is given and violated, how many times to try with a new randomization. Defaults to 10.
        tp_modes (tuple, optional): the ways (modes) in which to control for transition probabilities of syllables in the stream. Defaults to ("random", "word_structured", "position_controlled").
        require_all_tp_modes (bool, optional): all streams coming from the same lexicon will be discarded if not all their tp-modes have been found. Defaults to True.

    Returns:
        RegisterType: _description_
    """
    logger.info("Building streams from lexicons ...")

    streams = {}

    for i, lexicon in enumerate(lexicons):
        found_all_tp_modes = True
        new_streams = {}
        for tp_mode in tp_modes:

            maybe_stream: Optional[StreamType] = make_stream_from_lexicon(
                lexicon,
                max_rhythmicity=max_rhythmicity,
                max_tries_randomize=max_tries_randomize,
                n_repetitions=stream_length,
                tp_mode=tp_mode
            )

            if maybe_stream:
                new_streams[f"{''.join(word.id for word in lexicon)}_{tp_mode}"] = maybe_stream
            else:
                found_all_tp_modes = False
                break

        if found_all_tp_modes or (require_all_tp_modes == False):
            streams.update(new_streams)

    streams_reg = Register(**streams)

    streams_reg.info = {
        "tp_modes": tp_modes,
        "max_rhythmicity": max_rhythmicity,
        "max_tries_randomize": max_tries_randomize,
        "stream_length": stream_length,
        "require_all_tp_modes": require_all_tp_modes
    }

    return streams_reg
