%% Setup

% Clear the workspace
clc
clear
close all

% Load stimuli
project_dir = '/data/u_titone_thesis/PhD_Leipzig/01_Projects/01_Artificial_Lexicon';
stimuli_dir = [project_dir filesep '01_Stimuli' filesep '03_Streams'];
sound_files = dir(fullfile(stimuli_dir, 'TP*.wav'));

% Experimental parameters
fSampl = 44100;                                     % sampling frequency
nRepet = 720;                                       % number of syllables in one trial
nSylls = 0.3;                                       % length of a syllable (in seconds)
nDurat = nSylls * nRepet;                           % length of a trial (in seconds)
nSampl = ceil(nDurat * fSampl);                     % length of a trial (in samples)
n_FOIs = 2;                                         % number of frequencies of interest
nConds = 3;                                         % number of conditions
nChunk = nSampl;                                    % number of samples in a snippet (nSampl = full)
nFreqs = fSampl * (0:(nChunk/2)) / nChunk;          % number of frequency bins

% Split into conditions
StimSet = zeros(nConds, nSampl);                    % initialize StimSet
OrderID = [2 1 3];
for sFile = 1:length(sound_files)
    sName = sound_files(OrderID(sFile)).name;
    sFold = sound_files(OrderID(sFile)).folder;
    StimSet(sFile, :) = audioread([sFold filesep sName]);
end

%% Compute fft and plot

% Loop though conditions, compute FFT and plot
names = {"TP-random_position-random"; "TP-random_position-fixed"; "TP-structured"};
t = tiledlayout(1, nConds, "TileSpacing", "compact");
for iCond = 1:nConds
    trl_sound = abs(hilbert(StimSet(iCond, :)));
    trl_freqs = fft(trl_sound, nChunk);
    trl_power = abs(trl_freqs/ nChunk);
    trl_power = trl_power(1:nChunk/2+1);
    trl_power(2:end-1) = 2*trl_power(2:end-1);
    nexttile
    plot(nFreqs, trl_power, 'black-')
    xlim([0 5])
    ylim([0 0.03])
    xticks([1.11 2.22 3.33 4.44])
    title(strsplit(erase(names{iCond}, '.wav'), '_'), ...
        'FontSize', 12, 'FontWeight', 'normal', 'FontName', 'Arial')
    set(gca,'TickDir','out')
    set(gca,'box','off')
    if iCond ~= 1
        yticks([])
    end
end
title(t, 'Envelope spectra', 'FontSize', 14, 'FontWeight', 'bold', 'FontName', 'Arial')
xlabel(t, 'f (Hz)', 'FontSize', 12, 'FontWeight', 'normal', 'FontName', 'Arial')
ylabel(t, '|P(f)|', 'FontSize', 12, 'FontWeight', 'normal', 'FontName', 'Arial')
f_out = fullfile(project_dir, "03_Figures/titone_Spectra_results_v3.tiff");
exportgraphics(gcf, f_out, 'Resolution', 600)
