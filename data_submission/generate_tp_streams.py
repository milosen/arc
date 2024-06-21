#%%#################################################################################################
############################### TRANSITIONAL PROBABILITY: STATISTICS ###############################
####################################################################################################

# # IMPORT STREAMS FROM TITONE, MILOSEVIC & MEYER (2024) ARC PACKAGE
# indir = project_dir + '01_Stimuli/02_Lexicons/'
# fname = indir + 'best_lexicon.txt'
# fdata = list(csv.reader(open(fname, "r")))
# lexicon_words = fdata[0][0].split(": ")[1].split("|")
# lexicon_sylls = [list(map(''.join, zip(*[iter(i)]*nPoss))) for i in lexicon_words]
# TP_posrdm_arc = fdata[2][0].split(": ")[1].split("|")
# TP_struct_arc = fdata[7][0].split(": ")[1].split("|")
# TP_posfix_arc = fdata[12][0].split(": ")[1].split("|")
# TP_stream_arc = [TP_struct_arc, TP_posfix_arc, TP_posrdm_arc]
# TP_fnames_arc = ['TP_struct_ARC', 'TP_posfix_ARC', 'TP_posrdm_ARC']

# # COMPUTE INCREMENTAL TRANSITION MATRIX FOR EACH STREAM IN EACH METHOD
# PRW = []
# for trial in TP_stream_arc:
#     sylls = list(set(trial))
#     items = trial[:]
#     for iSyll in range(nSyll):
#         idxes = np.where(np.array(trial) == sylls[iSyll])[0].tolist()
#         for i_idx in idxes:
#             items[i_idx] = iSyll
#     PRW.append(items)
# SHF = []
# SHF.append(shuffled_random_stream(nSyll, nReps))
# SHF.append(shuffled_struct_srteam(nTrip, nSyll, nReps))
# Incremental_TPs = []
# for method in [PRW, SHF]:
#     R = []
#     for condition in method:
#         items = condition[:]
#         r = []
#         for iSyll in range(1, len(items)):
#             k = transitional_p_matrix(items[:iSyll+1])
#             r.append(k[items[iSyll-1]][items[iSyll]])
#         plt.scatter(range(len(r)), r)
#         R.append(r)
#     plt.show()
#     Incremental_TPs.append(R)

# # SAVE INCREMENTAL TP MATRIX FOR ALL STREAMS
# opdir = project_dir + '01_Stimuli/03_Streams/'
# fname = opdir + 'Incremental_TPs.pickle'
# fdata = Incremental_TPs
# with open(fname, 'wb') as f:
#     pickle.dump(fdata, f, pickle.HIGHEST_PROTOCOL)

# LOAD INCREMENTAL TP MATRIX FOR ALL STREAMS
indir = project_dir + '01_Stimuli/03_Streams/'
fname = indir + 'Incremental_TPs.pickle'
with open(fname, 'rb') as f:
   fdata = pickle.load(f)
Incremental_TPs = fdata
