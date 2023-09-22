import csv
import os
import pickle

import numpy as np
from scipy.stats import stats

from arc.phonecodes import phonecodes


def sample_lexicon_with_minimal_overlap(words, sylls, feats, ovlap):
    pass


def lexicons():
    # %%#################################################################################################
    ############################################# LEXICONS #############################################
    ####################################################################################################

    # # LOAD WORDS
    # indir = project_dir + '01_Stimuli/02_Words/'
    # fname = indir + 'words.pickle'
    # with open(fname, 'rb') as f:
    #     fdata = pickle.load(f)
    # Words = fdata[0]; Trips = fdata[1]

    # KERNELS SIMULATING A STATIONARY OSCILLATORY SIGNAL AT THE FREQUENCY OF INTEREST (LAG = 3)
    patts = [tuple([i for i in np.roll((1, 0, 0, 1, 0, 0), j)]) for j in range(nPoss)]

    # EXTRACT MATRIX OF BINARY FEATURES FOR EACH TRIPLET AND COMPUTE FEATURES OVERLAP FOR EACH PAIR
    Sylls = [[w[i:j] for i, j in zip(list(range(0, nPoss * nPoss, nPoss)),
                                     list(range(0, nPoss * nPoss, nPoss))[1:] + [None])]
             for w in Words]
    Feats = [binary_feature_matrix(i) for i in Sylls]
    Ovlap = compute_feats_overlap(Words, Feats, patts)

    # GENERATE SETS OF WORDS WITH UNIQUE SYLLABLES AND MINIMUM FEATURES OVERLAP ACROSS WORDS
    Sets_of_Words, Sets_of_Feats = generate_lexicon_sets(Words, Sylls, Feats, Ovlap)

    # # SAVE LEXICONS
    # opdir = project_dir + '01_Stimuli/03_Lexicons/'
    # fname = opdir + 'lexicons.pickle'
    # fdata = [Sets_of_Words, Sets_of_Feats]
    # with open(fname, 'wb') as f:
    #     pickle.dump(fdata, f, pickle.HIGHEST_PROTOCOL)

    # %%#################################################################################################
    ############################################## STREAM ##############################################
    ####################################################################################################

    # # LOAD LEXICONS
    # indir = project_dir + '01_Stimuli/03_Lexicons/'
    # fname = indir + 'lexicons.pickle'
    # with open(fname, 'rb') as f:
    #     fdata = pickle.load(f)
    # Sets_of_Words = fdata[0]; Sets_of_Feats = fdata[1]

    # # GENERATE PSEUDO-RANDOM STREAMS OF SYLLABLES CONTROLLING FOR TPs
    # TP_struct_V = []; TP_random_V = []
    # while len(TP_struct_V) < nRand:
    #     v_struct, M_struct = pseudo_rand_TP_struct()
    #     v_random, M_struct = pseudo_rand_TP_random()
    #     if v_struct not in TP_struct_V and v_random not in TP_random_V:
    #         TP_struct_V.append(v_struct)
    #         TP_random_V.append(v_random)

    # # SAVE RANDOMIZATIONS (nRand = 100000, nReps = 48)
    # opdir = project_dir + '01_Stimuli/04_Streams/'
    # fname = opdir + 'pseudo_rand.pickle'
    # fdata = [TP_struct_V, TP_random_V]
    # with open(fname, 'wb') as f:
    #     pickle.dump(fdata, f, pickle.HIGHEST_PROTOCOL)

    # LOAD RANDOMIZATIONS (nRand = 100000, nReps = 48)
    indir = project_dir + '01_Stimuli/04_Streams/'
    fname = indir + 'pseudo_rand.pickle'
    with open(fname, 'rb') as f:
        fdata = pickle.load(f)
    TP_struct_V = fdata[0];
    TP_random_V = fdata[1]

    # SELECT TP STRUCT STREAMS AND TP RANDOM STREAMS WITH MINIMUM RHYTHMICITY INDEX
    TP_struct_trial = [];
    RI_struct_trial = []
    TP_random_trial = [];
    RI_random_trial = []
    Rhythmicity_max = []
    for iSets in range(len(Sets_of_Words)):
        print(iSets)
        tp_struct_trl = [];
        ri_struct_trl = []
        tp_random_trl = [];
        ri_random_trl = []
        words = Sets_of_Words[iSets][:]
        sylls = list(it.chain.from_iterable(
            [[w[i:j] for i, j in zip(list(range(0, nPoss * nPoss, nPoss)),
                                     list(range(0, nPoss * nPoss, nPoss))[1:] + [None])]
             for w in words]))
        for iRand in range(nRand):
            tp_struct = list(it.chain.from_iterable(
                [[w[i:j] for i, j in zip(list(range(0, nPoss * nPoss, nPoss)),
                                         list(range(0, nPoss * nPoss, nPoss))[1:] + [None])]
                 for w in [words[i] for i in TP_struct_V[iRand]]]))
            tp_random = [sylls[i] for i in TP_random_V[iRand]]
            ri_struct = features_stationarity(tp_struct, patts)
            ri_random = features_stationarity(tp_random, patts)
            if max(ri_struct) <= 0.1:
                tp_struct_trl.append(tp_struct)
                ri_struct_trl.append(ri_struct)
            if max(ri_random) <= 0.1:
                tp_random_trl.append(tp_random)
                ri_random_trl.append(ri_random)
        tp_struct_idx = np.argsort([max(i) for i in ri_struct_trl])[:nTrls]
        TP_struct_trl = [tp_struct_trl[i] for i in tp_struct_idx]
        RI_struct_trl = [ri_struct_trl[i] for i in tp_struct_idx]
        tp_random_idx = np.argsort([max(i) for i in ri_random_trl])[:nTrls]
        TP_random_trl = [tp_random_trl[i] for i in tp_random_idx]
        RI_random_trl = [ri_random_trl[i] for i in tp_random_idx]
        ri_struct_max = max(max(RI_struct_trl))
        ri_random_max = max(max(RI_struct_trl))
        rhythmicity_max = max([ri_struct_max, ri_random_max])
        TP_struct_trial.append(TP_struct_trl)
        TP_random_trial.append(TP_random_trl)
        RI_struct_trial.append(RI_struct_trl)
        RI_random_trial.append(RI_random_trl)
        Rhythmicity_max.append(rhythmicity_max)
    best_RI = np.argsort(Rhythmicity_max)[:nSets]
    TP_struct_stream = [TP_struct_trial[i] for i in best_RI]
    TP_random_stream = [TP_random_trial[i] for i in best_RI]
    RI_struct_stream = [RI_struct_trial[i] for i in best_RI]
    RI_random_stream = [RI_random_trial[i] for i in best_RI]

    # SAVE LEXICON TRIALS
    opdir = project_dir + '01_Stimuli/04_Streams/'
    fname = indir + 'streams.pickle'
    fdata = [TP_struct_stream, TP_random_stream, RI_struct_stream, RI_random_stream]
    with open(fname, 'wb') as f:
        pickle.dump(fdata, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
