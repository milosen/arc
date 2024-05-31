####################################
##### Syllables pre-processing #####
####################################

# clear
clearinfo

# project directory
dir$ = "/data/u_titone_thesis/PhD_Leipzig/01_Projects/01_Artificial_Lexicon/01_Stimuli/"

# directories
input$ = dir$ + "01_Syllables/04_Audios/"
output$ = dir$ + "01_Syllables/06_Sounds/"

# set parameters
time_resolution = 0.001
amplitude_threshold = 0.01
padding = 0.00

# get files info
Create Strings as file list... list 'input$'/*.wav
number = Get number of strings

# loop
for i from 1 to number
	
	# read silence
	Read from file: dir$ + "01_Syllables/05_Silence/10ms_silence.wav"
	
	# read file
	select Strings list
	soundfile$ = Get string... i
	name$ = replace$ (soundfile$, ".wav", "", 1)
	fullfile$ = input$ + soundfile$
	Read from file... 'fullfile$'

	##################
	### Trim edges ###
	##################

	# get onset
	position = 0
	totaldur = Get total duration
	amplitude = Get value at time... 'position' Sinc700
	amplitude = abs(amplitude)
	if amplitude = undefined
		amplitude = 0
	endif
	while amplitude < amplitude_threshold	
   		position = 'position' + time_resolution
		amplitude = Get value at time... 'position' Sinc700
		amplitude = abs(amplitude)
		if amplitude = undefined
			amplitude = 0
		endif
	endwhile
	onset = position - padding

	# get offset
	position = totaldur
	amplitude = Get value at time... 'position' Sinc700
	amplitude = abs(amplitude)
	if amplitude = undefined
		amplitude = 0
	endif
	while amplitude < amplitude_threshold	
		position = 'position' - time_resolution
		amplitude = Get value at time... 'position' Sinc700
		amplitude = abs(amplitude)
		if amplitude = undefined
			amplitude = 0
		endif		
	endwhile
	offset = position + padding

	# cut silences
	appendInfoLine: onset, "  ", offset
	Extract part... onset offset rectangular 1.0 no
	select Sound 'name$'_part
	Edit
		editor Sound 'name$'_part
		Move cursor to... onset
		Move cursor to nearest zero crossing
		onset = Get cursor
		Move cursor to... offset
		Move cursor to nearest zero crossing
		offset = Get cursor
	endeditor
	
	# scale intensity
	Scale intensity: 65
	
	# smooth edge at onset
	plusObject: "Sound 10ms_silence"
	Concatenate with overlap: 0.01
	
	# remove objects
	selectObject:  "Sound 10ms_silence"
	Remove
	
	# smooth edge at offset
	Read from file: dir$ + "01_Syllables/05_Silence/10ms_silence.wav"
	selectObject: "Sound chain"
	plusObject: "Sound 10ms_silence"
	Concatenate with overlap: 0.01
	
	# rename
	Rename: name$ + "_trimmed"

	# remove objects
	selectObject:  "Sound chain"
	plusObject: "Sound " + name$ + "_part"
	Remove

	############################
	### Change duration tier ###
	############################

	# change duration tier (280 ms)
	length = 0.28
	selectObject: "Sound " + name$ + "_trimmed"
	totaldur = Get total duration
	coeffdur = length/totaldur
	To Manipulation: 0.01, 75, 600
	Extract duration tier
	Add point: 0, coeffdur
	Add point: totaldur, coeffdur
	selectObject: "Manipulation " + name$ + "_trimmed"
	plusObject: "DurationTier " + name$ + "_trimmed"
	Replace duration tier
	selectObject: "Manipulation " + name$ + "_trimmed"
	Get resynthesis (overlap-add)
	Rename: name$ + "_280ms"
	
	# append silence at onset
	selectObject: "Sound " + name$ + "_280ms"
	plusObject: "Sound 10ms_silence"
	Concatenate
	Rename: name$ + "_10ms_280ms"
	
	# remove objects
	selectObject: "Sound " + name$ + "_280ms"
	plusObject: "DurationTier " + name$ + "_trimmed"
	plusObject: "Manipulation " + name$ + "_trimmed"
	plusObject: "Sound 10ms_silence"
	Remove
	
	# read silence
	Read from file: dir$ + "01_Syllables/05_Silence/10ms_silence.wav"

	# append silence at offset
	selectObject: "Sound " + name$ + "_10ms_280ms"
	plusObject: "Sound 10ms_silence"
	Concatenate
	Rename: name$ + "_10ms_280ms_10ms"
	
	###########################
	### Save processed file ###
	###########################
	
	# save output
	# Save as WAV file... 'output$'/'soundfile$'
	
	#####################
	### Clear objects ###
	#####################
	
	# remove objects
	selectObject:  "Sound " + name$
	plusObject: "Sound 10ms_silence"
	Remove

endfor
