import mido
import pedalboard as pb
from mido import Message
from pprint import pprint


notes = [
    'A',
    'A#',
    'B',
    'C',
    'C#',
    'D',
    'D#',
    'E',
    'F',
    'F#',
    'G',
    'G#'
]

octave_range = [0, 8]

# A0 = 21
# G9 = 127
midi_notes_all = range(21,128)

# A0 = 21
# C8 = 108
piano_midi_notes_all = range(21,109)

sample_rate = 48000

num_channels = 2

# normal MIDI is 24
# high PPQN is 480 or 960
# Ableton Live exports use 96
# mido_default is 480
midi_tick_rate = 480

tempos = [
    60,
    90,
    120,
    150,
    180
]

time_signatures = [
    '2/2',
    '3/2',
    '4/2',
    '2/4',
    '3/4',
    '4/4',
    '5/4',
    '6/4',
    '7/4',
    '8/4',
    '3/8',
    '4/8',
    '5/8',
    '6/8',
    '7/8',
    '8/8',
    '9/8',
    '12/8'
]

note_lengths = [
    '1/32',
    '1/16',
    '1/8',
    '1/4',
    '1/2',
    '1/1'
]

loop_lengths_in_bars = [
    1,
    2,
    4,
    8
]


serum2 = pb.load_plugin('Serum2.vst3', plugin_name='Serum 2')


for loop_length_in_bars in loop_lengths_in_bars:
    for tempo in tempos:
        for time_signature in time_signatures:
            for note_length in note_lengths:
                print(f"Loop length = {loop_length_in_bars} bars")
                print(f"Tempo = {tempo}")
                print(f"Time signature = {time_signature}")
                # some note lengths may not make sense in certain time signatures
                numerator, denominator = map(int, time_signature.split('/'))
                if denominator / int(note_length.split('/')[1]) / numerator > 1:
                    print(f"Skipping because {denominator} divided by {int(note_length.split('/')[1])} divided by {numerator} is > 1.0")
                    continue
                # skip if the resulting loop would be equivalent to a one-shot
                if denominator / int(note_length.split('/')[1]) / numerator == 1 and loop_length_in_bars == 1:
                    print(f"Skipping because {denominator} divided by {int(note_length.split('/')[1])} divided by {numerator} is just a one-shot if loop length is {loop_length_in_bars}")
                    continue
                print(f"{numerator} * 1/{denominator} notes per bar")
                print(f"Note length = {note_length}")
                # fun fact:
                # the correct way to write a note that lasts for a full 9/8 bar is to tie a dotted minim (dotted half-note) to a dotted crotchet (dotted quarter-note).
                # 1.5 * 1/2 = 0.5 + 0.25 = dotted minim (dotted half-note)
                # 1.5 * 1/4 = 0.25 + 0.125 = dotted crotchet (dotted quarter-note)
                # Total = 1.125
                # A "whole note" in 9/8 would be 8 * 8th notes.
                notes_per_bar = int((int(note_length.split('/')[1]) / denominator) * numerator)
                print(f"{notes_per_bar} * {note_length} notes per bar")
                midi_tempo = mido.bpm2tempo(tempo, time_signature=(numerator, denominator))
                # 480 ticks per 1/4 note
                # 240 ticks per 1/8 note
                # 960 ticks per 1/2 note
                note_length_in_ticks = 480 * (4 / int(note_length.split('/')[1]))
                print(f"note length in ticks = {note_length_in_ticks}")
                note_length_in_seconds = mido.tick2second(note_length_in_ticks, 480, midi_tempo)
                print(f"note length in seconds = {note_length_in_seconds}")
                midi_sequence = []
                for index, midi_note_number in enumerate(piano_midi_notes_all):
                    midi_sequence.append(Message("note_on", note=midi_note_number, time=note_length_in_seconds * index * 2))
                    midi_sequence.append(Message("note_off", note=midi_note_number, time=note_length_in_seconds * index * 2 + 1))
                midi_sequence_duration = len(midi_sequence) * note_length_in_seconds
                print(f"MIDI sequence duration = {midi_sequence_duration} seconds")
                audio = serum2(midi_messages=midi_sequence, duration=midi_sequence_duration, sample_rate=sample_rate, num_channels=num_channels)
                with pb.io.AudioFile(f"Serum2_A0-C8_{tempo}BPM_{int(note_length.split('/')[1])}notes_{numerator}n{denominator}d.wav", "w", sample_rate, num_channels=num_channels) as f:
                    f.write(audio)


import os

for tempo in [60, 70, 80, 90, 100, 110, 120, 128, 135, 140, 150, 160, 175, 180, 200]:
    for time_signature in ['4/4']:
        for note_length in note_lengths:
            print(f"Tempo = {tempo}")
            numerator, denominator = map(int, time_signature.split('/'))
            print(f"Time Signature = {time_signature} = {numerator} * 1/{denominator} notes per bar")
            # some note lengths may not make sense in certain time signatures
            if denominator / int(note_length.split('/')[1]) / numerator > 1:
                print(f"Skipping because {denominator} divided by {int(note_length.split('/')[1])} divided by {numerator} is > 1.0")
                continue
            # fun fact:
            # the correct way to write a note that lasts for a full 9/8 bar is to tie a dotted minim (dotted half-note) to a dotted crotchet (dotted quarter-note).
            # 1.5 * 1/2 = 0.5 + 0.25 = dotted minim (dotted half-note)
            # 1.5 * 1/4 = 0.25 + 0.125 = dotted crotchet (dotted quarter-note)
            # Total = 1.125
            # A "whole note" in 9/8 would be 8 * 8th notes.
            notes_per_bar = int((int(note_length.split('/')[1]) / denominator) * numerator)
            print(f"Note length = {note_length} => {notes_per_bar} * {note_length} notes per bar")
            midi_tempo = mido.bpm2tempo(tempo, time_signature=(numerator, denominator))
            # 480 ticks per 1/4 note
            # 240 ticks per 1/8 note
            # 960 ticks per 1/2 note
            note_length_in_ticks = 480 * (4 / int(note_length.split('/')[1]))
            print(f"note length in ticks = {note_length_in_ticks}")
            note_length_in_seconds = mido.tick2second(note_length_in_ticks, 480, midi_tempo)
            print(f"note length in seconds = {note_length_in_seconds}")
            midi_sequence = []
            for index, midi_note_number in enumerate(piano_midi_notes_all):
                midi_sequence.append(Message("note_on", note=midi_note_number, time=note_length_in_seconds * index * 2))
                midi_sequence.append(Message("note_off", note=midi_note_number, time=note_length_in_seconds * (index * 2 + 1)))
            midi_sequence_duration = (len(midi_sequence) + 1) * note_length_in_seconds
            print(f"MIDI sequence duration = {midi_sequence_duration} seconds")
            audio = serum2(midi_messages=midi_sequence, duration=midi_sequence_duration, sample_rate=sample_rate, num_channels=num_channels)
            preset_name = 'Sine_256atk_256rel'
            folder_name = f"Serum2_{preset_name}_A0-C8"
            if not os.path.exists(os.path.join(os.getcwd(), 'renders', folder_name)):
                os.makedirs(os.path.join(os.getcwd(), 'renders', folder_name), exist_ok=True)
            render_filename = f"Serum2_{preset_name}_A0-C8_{tempo}BPM_{int(note_length.split('/')[1])}notes_{numerator}n{denominator}d.wav"
            file_path = os.path.join(os.getcwd(), 'renders', folder_name, render_filename)
            with pb.io.AudioFile(file_path, "w", sample_rate, num_channels=num_channels, bit_depth=32) as f:
                f.write(audio)

#with AudioStream(output_device_name=r"Speakers (JadeAudio JA11(2.0))") as stream:
#    stream.play(audio=audio, sample_rate=sample_rate, output_device_name=r"Speakers (JadeAudio JA11(2.0))")

                for midi_note_number in piano_midi_notes_all:
...                 print(f"MIDI note number = {midi_note_number}")
...                 midi_sequence = []
...                 for bar_index in range(bars):
...                     print(f"Generating bar {bar_index + 1} of {bars}...")
...                     for note_index in range(notes_per_bar):
...                         if note_index % 2 == 0:
...                             midi_sequence.append(Message("note_on", note=midi_note_number, time=note_length_in_seconds * (note_index + (bar_index * notes_per_bar))))
...                         else:
...                             midi_sequence.append(Message("note_off", note=midi_note_number, time=note_length_in_seconds * (note_index + (bar_index * notes_per_bar))))

0 bar
    index 0 * 0.25 = 0      on
    index 1 * 0.25 = 0.25   off
    index 2 * 0.25 = 0.5    on
    index 3 * 0.25 = 0.75   off
1 bar
    index 4 * 0.25 = 1.0    on
    index 5 * 0.25 = 1.25   off
    index 6 * 0.25 = 1.5    on
    index 7 * 0.25 = 1.75   off
2 bar
    index 8 * 0.25 = 2.0    on
    index 9 * 0.25 = 2.25   off
    index 10 * 0.25 = 2.5   on
    index 11 * 0.25 = 2.75  off
3 bar


4 bars
    4x 1/4 notes per bar in 4/4
        on 0 = 0s
        off note_length_in_seconds = 0.5s
        on note_length_in_seconds * 2 = 1.0s
        off note_length_in_seconds * 3 = 1.5s
    6x 1/8 notes per bar in 3/4
        on 0
        off note_length_in_seconds
        on note_length_in_seconds * 2
        off note_length_in_seconds * 3
        on ote_length_in_seconds * 4
        off note_length_in_seconds * 5
    9x 1/8 notes per bar in 9/8
        on 0
        off note_length_in_seconds = 0.25s
        on note_length_in_seconds * 2 = 0.5s
        off note_length_in_seconds * 3 = 0.75s
        on ote_length_in_seconds * 4 = 1.0s
        off note_length_in_seconds * 5 = 1.25s
        on note_length_in_seconds * 6 = 1.5s
        off note_length_in_seconds * 7 = 1.75s
        on ote_length_in_seconds * 8 = 2.0s
        off note_length_in_seconds * 9 = 2.25s




import json
import os
import mido
import pedalboard as pb
from pprint import pprint
from mido import Message
from pedalboard._pedalboard import WrappedBool

# A0 = 21
# C8 = 108
piano_midi_notes_all = range(21,109)

sample_rate = 48000

num_channels = 2

note_lengths = [
    '1/32',
    '1/16',
    '1/8',
    '1/4',
    '1/2',
    '1/1'
]

loop_lengths_in_bars = [
    1,
    2,
    4,
    8
]

# 60 75 90 105 120 135 150 165 180 195 210
tempos = [tempo for tempo in range(60, 210, 15)]

tempos = [60, 70, 80, 90, 100, 110, 120, 128, 135, 140, 150, 160, 175, 180, 200]

serum2 = pb.load_plugin('Serum2.vst3', plugin_name='Serum 2') 

for tempo in [60, 70, 80, 90, 100, 110, 120, 128, 135, 140, 150, 160, 175, 180, 200]:
    for time_signature in ['3/4', '4/4', '9/8']:
        for note_length in note_lengths:
            print(f"Tempo = {tempo}")
            numerator, denominator = map(int, time_signature.split('/'))
            print(f"Time Signature = {time_signature} = {numerator} * 1/{denominator} notes per bar")
            # some note lengths may not make sense in certain time signatures
            if denominator / int(note_length.split('/')[1]) / numerator > 1:
                print(f"Skipping because {denominator} divided by {int(note_length.split('/')[1])} divided by {numerator} is > 1.0")
                continue
            # fun fact:
            # the correct way to write a note that lasts for a full 9/8 bar is to tie a dotted minim (dotted half-note) to a dotted crotchet (dotted quarter-note).
            # 1.5 * 1/2 = 0.5 + 0.25 = dotted minim (dotted half-note)
            # 1.5 * 1/4 = 0.25 + 0.125 = dotted crotchet (dotted quarter-note)
            # Total = 1.125
            # A "whole note" in 9/8 would be 8 * 8th notes.
            notes_per_bar = int((int(note_length.split('/')[1]) / denominator) * numerator)
            print(f"Note length = {note_length} => {notes_per_bar} * {note_length} notes per bar")
            midi_tempo = mido.bpm2tempo(tempo, time_signature=(numerator, denominator))
            # 480 ticks per 1/4 note
            # 240 ticks per 1/8 note
            # 960 ticks per 1/2 note
            note_length_in_ticks = 480 * (4 / int(note_length.split('/')[1]))
            print(f"note length in ticks = {note_length_in_ticks}")
            note_length_in_seconds = mido.tick2second(note_length_in_ticks, 480, midi_tempo)
            print(f"note length in seconds = {note_length_in_seconds}")
            for midi_note_number in piano_midi_notes_all:
                print(f"MIDI note number = {midi_note_number}")
                midi_sequence = []
                for bar_index in range(bars):
                    print(f"Generating bar {bar_index + 1} of {bars}...")
                    for note_index in range(notes_per_bar):
                        midi_sequence.append(Message("note_on", note=midi_note_number, time=note_length_in_seconds * (note_index + (bar_index * notes_per_bar))))
                        midi_sequence.append(Message("note_off", note=midi_note_number, time=note_length_in_seconds * (note_index + 1 + (bar_index * notes_per_bar))))
                midi_sequence_duration = (len(midi_sequence)/2) * note_length_in_seconds
                print(f"MIDI sequence duration = {midi_sequence_duration} seconds")
                audio = serum2(midi_messages=midi_sequence, duration=midi_sequence_duration, sample_rate=sample_rate, num_channels=num_channels)
                preset_name = 'Square_256atk_256rel'
                folder_name = f"Serum2_{preset_name}_{midi_note_number}_{bars}bars"
                if not os.path.exists(os.path.join(os.getcwd(), 'renders', folder_name)):
                    os.makedirs(os.path.join(os.getcwd(), 'renders', folder_name), exist_ok=True)
                render_filename = f"Serum2_{preset_name}_{midi_note_number}_{bars}bars_{tempo}BPM_{int(note_length.split('/')[1])}notes_{numerator}n{denominator}d.wav"
                file_path = os.path.join(os.getcwd(), 'renders', folder_name, render_filename)
                with pb.io.AudioFile(file_path, "w", sample_rate, num_channels=num_channels, bit_depth=32) as f:
                    f.write(audio)




import json
import os
import mido
import pedalboard as pb
from pprint import pprint
from mido import Message
from pedalboard._pedalboard import WrappedBool

# A0 = 21
# C8 = 108
piano_midi_notes_all = range(21,109)

sample_rate = 48000

num_channels = 2

# 60 75 90 105 120 135 150 165 180 195
tempos = [tempo for tempo in range(60, 210, 15)]

time_signatures = [
    '2/2',
    '3/2',
    '4/2',
    '2/4',
    '3/4',
    '4/4',
    '5/4',
    '6/4',
    '7/4',
    '8/4',
    '3/8',
    '4/8',
    '5/8',
    '6/8',
    '7/8',
    '8/8',
    '9/8',
    '12/8'
]

bars = 4

note_lengths = [
    '1/32',
    '1/16',
    '1/8',
    '1/4',
    '1/2',
    '1/1'
]

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
octaves = list(range(11))
notes_per_octave = len(note_names)


def number_to_note(number: int) -> tuple:
    note_name = note_names[number % notes_per_octave]
    note_octave = (number // notes_per_octave) - 1
    # F#0, C3, G9, etc.
    return ''.join([note_name, str(note_octave)])


def note_to_number(note: str) -> int:
    octave = int(note[-1]) + 1
    note_number = note_names.index(note[0:-1])
    note_number += (notes_per_octave * octave)
    return note_number


serum2 = pb.load_plugin('Serum2.vst3', plugin_name='Serum 2') 

preset_name = 'Serum2_SawUp_halfatk_halfrel_0phase_0rand_mono'

for tempo in tempos:
    for time_signature in time_signatures:
        for note_length in note_lengths:
            print(f"Tempo = {tempo}")
            numerator, denominator = map(int, time_signature.split('/'))
            print(f"Time Signature = {time_signature} = {numerator} * 1/{denominator} notes per bar")
            # some note lengths may not make sense in certain time signatures
            if denominator / int(note_length.split('/')[1]) / numerator > 1:
                print(f"Skipping because {denominator} divided by {int(note_length.split('/')[1])} divided by {numerator} is > 1.0")
                continue
            # fun fact:
            # the correct way to write a note that lasts for a full 9/8 bar is to tie a dotted minim (dotted half-note) to a dotted crotchet (dotted quarter-note).
            # 1.5 * 1/2 = 0.5 + 0.25 = dotted minim (dotted half-note)
            # 1.5 * 1/4 = 0.25 + 0.125 = dotted crotchet (dotted quarter-note)
            # Total = 1.125
            # A "whole note" in 9/8 would be 8 * 8th notes.
            notes_per_bar = int((int(note_length.split('/')[1]) / denominator) * numerator)
            print(f"Note length = {note_length} => {notes_per_bar} * {note_length} notes per bar")
            midi_tempo = mido.bpm2tempo(tempo, time_signature=(numerator, denominator))
            # 480 ticks per 1/4 note
            # 240 ticks per 1/8 note
            # 960 ticks per 1/2 note
            note_length_in_ticks = 480 * (4 / int(note_length.split('/')[1]))
            print(f"note length in ticks = {note_length_in_ticks}")
            note_length_in_seconds = mido.tick2second(note_length_in_ticks, 480, midi_tempo)
            print(f"note length in seconds = {note_length_in_seconds}")
            for midi_note_number in piano_midi_notes_all:
                print(f"MIDI note number = {midi_note_number}")
                midi_sequence = []
                for bar_index in range(bars):
                    print(f"Generating bar {bar_index + 1} of {bars}...")
                    for note_index in range(notes_per_bar):
                        if note_index % 2 == 0:
                            midi_sequence.append(Message("note_on", note=midi_note_number, time=note_length_in_seconds * (note_index + (bar_index * notes_per_bar))))
                        else:
                            midi_sequence.append(Message("note_off", note=midi_note_number, time=note_length_in_seconds * (note_index + (bar_index * notes_per_bar))))
                midi_sequence_duration = len(midi_sequence) * note_length_in_seconds
                print(f"MIDI sequence duration = {midi_sequence_duration} seconds")
                attack_time = ''.join(['1/', str(int(note_length.split('/')[1]) * 2)])
                release_time = attack_time
                serum2.env_1_attack = attack_time
                serum2.env_1_release = release_time
                audio = serum2(midi_messages=midi_sequence, duration=midi_sequence_duration, sample_rate=sample_rate, num_channels=num_channels)
                folder_name = f"{preset_name}_{number_to_note(midi_note_number)}_{bars}bars"
                if not os.path.exists(os.path.join(os.getcwd(), 'renders', folder_name)):
                    os.makedirs(os.path.join(os.getcwd(), 'renders', folder_name), exist_ok=True)
                render_filename = f"{folder_name}_{tempo}BPM_alternating{int(note_length.split('/')[1])}notes_{numerator}n{denominator}d.wav"
                file_path = os.path.join(os.getcwd(), 'renders', folder_name, render_filename)
                with pb.io.AudioFile(file_path, "w", sample_rate, num_channels=num_channels, bit_depth=32) as f:
                    f.write(audio)




param_value_dict = {parameter_name: getattr(serum2, parameter_name) for parameter_name in serum2.parameters.keys()}
from pedalboard._pedalboard import WrappedBool
param_value_dict = {k: (bool(v) if isinstance(v, WrappedBool) else v) for k, v in param_value_dict.items()}

with open('serum2_init_params.bin', 'wb') as file:
    file.write(serum2.raw_state)

try:
    with open('serum2_init_params.bin', 'rb') as file:
        serum2.raw_state = file.read()
except FileNotFoundError:
    print(f"serum2_init_params.bin not found!")
    pass




# manually tweak a patch/preset in the GUI
serum2.show_editor()
param_value_dict = {parameter_name: getattr(serum2, parameter_name) for parameter_name in serum2.parameters.keys()}
# convert pedalboard._pedalboard.WrappedBool to True/False
param_value_dict = {k: (bool(v) if isinstance(v, WrappedBool) else v) for k, v in param_value_dict.items()}
preset_name = 'SawUp_halfatk_halfrel_0phase_0rand_mono'
# save JSON
with open(f"{preset_name}_params.json", 'w') as file:
    json.dump(param_value_dict, file, indent=4, ensure_ascii=True)
# save raw state
with open(f"{preset_name}_params.bin", 'wb') as file:
    file.write(serum2.raw_state)
# set dict
for parameter_name, serialized_value in param_value_dict.items():
    setattr(serum2, parameter_name, serialized_value)
# load JSON
with open(f"{preset_name}_params.json", "r") as file:
    param_dict_from_json = json.load(file)
# set from JSON
for parameter_name, serialized_value in param_dict_from_json.items():
    setattr(serum2, parameter_name, serialized_value)
# load from raw state binary file and set raw state
try:
    with open(f"{preset_name}_params.bin", 'rb') as file:
        serum2.raw_state = file.read()
except FileNotFoundError:
    print(f"{preset_name}_params.bin not found! Skipping...")
    pass
# Test if loaded JSON dict is different from the in-memory dict
{key for key in param_value_dict.keys() if param_value_dict[key] != param_dict_from_json[key]}




for tempo in tempos:
    for time_signature in time_signatures:
        for note_length in note_lengths:
            print(f"Tempo = {tempo}")
            numerator, denominator = map(int, time_signature.split('/'))
            print(f"Time Signature = {time_signature} = {numerator} * 1/{denominator} notes per bar")
            # some note lengths may not make sense in certain time signatures
            # if 1/4 note in 3/4, then 4 / 4 / 3, ergo 1 / 3, which is < 1
            # if 1/1 note in 9/8, then 8 / 1 / 9, ergo 8/9, which is < 1
            # if 1/1 note in 3/4, then 4 / 1 / 3, ergo 4 / 3, which is > 1, so skip it
            if denominator / int(note_length.split('/')[1]) / numerator > 1:
                print(f"Skipping because {denominator} divided by {int(note_length.split('/')[1])} divided by {numerator} is > 1.0")
                continue
            # fun fact:
            # the correct way to write a note that lasts for a full 9/8 bar is to tie a dotted minim (dotted half-note) to a dotted crotchet (dotted quarter-note).
            # 1.5 * 1/2 = 0.5 + 0.25 = dotted minim (dotted half-note)
            # 1.5 * 1/4 = 0.25 + 0.125 = dotted crotchet (dotted quarter-note)
            # Total = 1.125
            # A "whole note" in 9/8 would be 8 * 8th notes.
            #
            # 1/1 note in 6/4 => 1x 1/1 note per bar
            # range(notes_per_bar) => [0] => 1 item in list, with item value == 0
            # note_index % 2 == 0 => 0 % 2 == 0 => True => note on @ time == 0
            notes_per_bar = (int(note_length.split('/')[1]) / denominator) * numerator
            note_ratio = (numerator / denominator) / (int(note_length.split('/')[0]) / int(note_length.split('/')[1]))
            if (note_ratio) != int(note_ratio):
                print(f"Skipping because note ratio was not a nice round number: {note_ratio}. I can't be arsed to cook up the logic for alterating 1/2 notes in 3/8 :B")
                continue
            print(f"Note length = {note_length} => {notes_per_bar} * {note_length} notes per bar")
            midi_tempo = mido.bpm2tempo(tempo, time_signature=(numerator, denominator))
            # 480 ticks per 1/4 note
            # 240 ticks per 1/8 note
            # 960 ticks per 1/2 note
            note_length_in_ticks = 480 * (4 / int(note_length.split('/')[1]))
            print(f"note length in ticks = {note_length_in_ticks}")
            note_length_in_seconds = mido.tick2second(note_length_in_ticks, 480, midi_tempo)
            print(f"note length in seconds = {note_length_in_seconds}")
            for midi_note_number in piano_midi_notes_all:
                print(f"MIDI note number = {midi_note_number}")
                midi_sequence = []
                for bar_index in range(bars):
                    print(f"Generating bar {bar_index + 1} of {bars}...")
                    for note_index in range(notes_per_bar):
                        if note_index % 2 == 0:
                            midi_sequence.append(Message("note_on", note=midi_note_number, time=note_length_in_seconds * (note_index + (bar_index * notes_per_bar))))
                            # 1/1 note in 6/4 => 1x 1/1 note per bar
                            # range(notes_per_bar) => [0] => 1 item in list, with item value == 0
                            # note_index % 2 == 0 => 0 % 2 == 0 => True => note on @ time == 0
                            #
                            # if 1/1 * 4/4 > ((6/4) / 2) then append a note-off at 4/4 (1/1) out of 6/4
                            # 1 > ((6/4) / 2) = 1 > 1.5 / 2 = 1 > 0.75 = True
                            # 1/8 * 4/4 > ((6/4) / 2) = 0.125 > 1.5 / 2 = 1 > 0.75 = False
                            # 1/8 * 8/8 > ((7/8) / 2) = 0.125 > 0.875 / 2 = 0.125 > 0.4375 = False
                            # 1/2 * 8/8 > ((9/8) / 2) = 0.5 > 0.5625 = False
                            #
                            # 1/8 > ((3/8) / 2) = 0.125 > 0.1875 = False
                            # 0   1   2   0   1   2   0   1   2   0   1   2
                            # on  off on  on  off on  on  off on  on  off on
                            #
                            # notes_per_bar = int((int(note_length.split('/')[1]) / denominator) * numerator)
                            # 2 / 8 * 9 = 2.25
                            # int(2.25) = 2
                            if int(note_length.split('/')[0]) / int(note_length.split('/')[1]) > ((numerator / denominator) / 2):
                                midi_sequence.append(Message("note_off", note=midi_note_number, time=note_length_in_seconds * (note_index + 1 + (bar_index * notes_per_bar))))
                            # 1/8 > ((3/8) / 2) = 0.125 > 0.1875 = False
                            # 0   1   2   0   1   2   0   1   2   0   1   2
                            # on  off on  on  off on  on  off on  on  off on
                            #
                            # 0   1   2   3   4   5   6   7   8   9   10  11 
                            # on  off on  off on  off on  off on  off on  off
                            #
                            # 1/2 in 9/8
                            # if 2 % 2 != 0 and 2 == 2 (first one is False)
                            # 0   1   0   1   0   1   0   1
                            # on  off on  off on  off on  off
                            #
                            #  0   1   2   3   4   5   6   7
                            #  on  off on  off on  off on  off
                            #
                            # 1/8 in 9/8
                            # 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35
                            # on  off on  off on  off on  off on off  on  off on  off on  off on off  on  off on off  on  off on  off on  off on  off on  off on  off on  off
                            #
                            # 1/8 in 3/8
                            # if 3 % 2 != 0 and 3 == 3 (both are True)
                            if ((note_index + 1) % 2 != 0) and ((note_index + 1) == notes_per_bar):
                                midi_sequence.append(Message("note_off", note=midi_note_number, time=note_length_in_seconds * (note_index + 1 + (bar_index * notes_per_bar))))
                        else:
                            midi_sequence.append(Message("note_off", note=midi_note_number, time=note_length_in_seconds * (note_index + (bar_index * notes_per_bar))))
                unique_midi_sequence = []
                [unique_midi_sequence.append(mess) for mess in midi_sequence if mess not in unique_midi_sequence]
                midi_sequence = unique_midi_sequence
                midi_sequence_duration = len(midi_sequence) * note_length_in_seconds
                print(f"MIDI sequence duration = {midi_sequence_duration} seconds")
                # 1/4 notes with ADSR_time_divisor == 2 will have attack/release/ times of 1/8 note, for example
                # Ergo it will take 1/8 note, or 1/2 of the 1/4 note, for the attack curve to max out,
                # and it will take another 1/8 note for the release time, so the 1/4 note's full audio length will be:
                # 1/8 attack + 1/8 hold and then decay to sustain level + 1/8 release, for a total length of 3/8 notes, 1/8 longer than 1/4 note.
                # 1/1 note in 6/4 = 1/4 attack + 3/4 hold + 1/2 release = 6/4 total time
                ADSR_time_divisor = 4
                attack_time = ''.join(['1/', str(int(note_length.split('/')[1]) * ADSR_time_divisor)])
                hold_time = attack_time
                decay_time = attack_time
                sustain_level = '-6.0 dB'
                release_time = attack_time
                decay_time = attack_time
                serum2.env_1_attack = attack_time
                serum2.env_1_hold = hold_time
                serum2.env_1_decay = decay_time
                serum2.env_1_sustain = sustain_level
                serum2.env_1_release = release_time
                serum2.env_1_atk_curve = 67.0
                serum2.env_1_dec_curve = 67.0
                serum2.env_1_rel_curve = 67.0
                print(f"ADSR => {serum2.env_1_attack} attack {serum2.env_1_hold} hold {serum2.env_1_decay} decay {serum2.env_1_sustain} sustain {serum2.env_1_release} release")
                serum2.a_unison = 2.0
                serum2.a_uni_detune = 0.25
                serum2.a_uni_blend = 75.0
                serum2.a_uni_width = 100.0
                print(f"Detune => {serum2.a_unison} voices {serum2.a_uni_detune} detune {serum2.a_uni_blend} blend {serum2.a_uni_width} width")
                audio = serum2(midi_messages=midi_sequence, duration=midi_sequence_duration, sample_rate=sample_rate, num_channels=num_channels)
                folder_name = f"{preset_name}_{number_to_note(midi_note_number)}_{bars}bars"
                if not os.path.exists(os.path.join(os.getcwd(), 'renders', '4bars', folder_name)):
                    os.makedirs(os.path.join(os.getcwd(), 'renders', '4bars', folder_name), exist_ok=True)
                render_filename = f"{folder_name}_{tempo}BPM_alternating{int(note_length.split('/')[1])}notes_{numerator}n{denominator}d.wav"
                file_path = os.path.join(os.getcwd(), 'renders', '4bars', folder_name, render_filename)
                with pb.io.AudioFile(file_path, "w", sample_rate, num_channels=num_channels, bit_depth=32) as f:
                    f.write(audio)




from pprint import pprint

# Example usage:
# Assuming synth_plugin.parameters['verb_wet'] is your parameter:
#   verb_wet = synth_plugin.parameters['verb_wet']
#   print_parameter_properties(verb_wet)
def print_parameter_properties(parameter):
    """
    Helper function to print out properties of an AudioProcessorParameter.

    Args:
    - parameter: An instance of AudioProcessorParameter.

    See: https://github.com/spotify/pedalboard/blob/f2c2ccd64e78abaf9b87bc2c59097965c8b92fe5/pedalboard/ExternalPlugin.h#L1307-L1310
    """
    # List of property names to display
    properties = [
        'index',
        'name',
        'python_name',
        'string_value',
        'raw_value',
        'default_raw_value',
        'range',
        'max_value',
        'min_value',
        'step_size',
        'approximate_step_size',
        'num_steps',
        'type',
        'units',
        'label',
        'is_discrete',
        'is_boolean',
        'is_orientation_inverted',
        'is_automatable',
        'is_meta_parameter',
    ]

    # Iterate over the property names
    for property_name in properties:
        try:
            # Access the property value
            property_value = getattr(parameter, property_name)
            # Print the property name and its value
            pprint(f"{property_name}: {property_value}")
        except AttributeError as e:
            # If the property does not exist, print an error message
            pprint(f"{property_name}: Property does not exist. Error: {e}")




import json
import os
import random
import re
import mido
import pedalboard as pb
from pprint import pprint
from mido import Message
from pedalboard._pedalboard import WrappedBool

# A0 = 21
# C8 = 108
piano_midi_notes_all = range(21,109)

sample_rate = 48000

num_channels = 2

# 60 75 90 105 120 135 150 165 180 195
tempos = [tempo for tempo in range(60, 210, 15)]

time_signatures = [
    '2/2',
    '3/2',
    '4/2',
    '2/4',
    '3/4',
    '4/4',
    '5/4',
    '6/4',
    '7/4',
    '8/4',
    '3/8',
    '4/8',
    '5/8',
    '6/8',
    '7/8',
    '8/8',
    '9/8',
    '12/8'
]

note_lengths = [
    '1/32',
    '1/16',
    '1/8',
    '1/4'
]

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
octaves = list(range(11))
notes_per_octave = len(note_names)


def number_to_note(number: int) -> tuple:
    note_name = note_names[number % notes_per_octave]
    note_octave = (number // notes_per_octave) - 1
    # F#0, C3, G9, etc.
    return ''.join([note_name, str(note_octave)])


def note_to_number(note: str) -> int:
    octave = int(note[-1]) + 1
    note_number = note_names.index(note[0:-1])
    note_number += (notes_per_octave * octave)
    return note_number


serum2 = pb.load_plugin('Serum2.vst3', plugin_name='Serum 2') 

bars = 2

for tempo in tempos:
    for time_signature in time_signatures:
        for note_length in note_lengths:
            print(f"Tempo = {tempo}")
            numerator, denominator = map(int, time_signature.split('/'))
            print(f"Time Signature = {time_signature} = {numerator} * 1/{denominator} notes per bar")
            # some note lengths may not make sense in certain time signatures
            # if 1/4 note in 3/4, then 4 / 4 / 3, ergo 1 / 3, which is < 1
            # if 1/1 note in 9/8, then 8 / 1 / 9, ergo 8/9, which is < 1
            # if 1/1 note in 3/4, then 4 / 1 / 3, ergo 4 / 3, which is > 1, so skip it
            if denominator / int(note_length.split('/')[1]) / numerator > 1:
                print(f"Skipping because {denominator} divided by {int(note_length.split('/')[1])} divided by {numerator} is > 1.0")
                continue
            # fun fact:
            # the correct way to write a note that lasts for a full 9/8 bar is to tie a dotted minim (dotted half-note) to a dotted crotchet (dotted quarter-note).
            # 1.5 * 1/2 = 0.5 + 0.25 = dotted minim (dotted half-note)
            # 1.5 * 1/4 = 0.25 + 0.125 = dotted crotchet (dotted quarter-note)
            # Total = 1.125
            # A "whole note" in 9/8 would be 8 * 8th notes.
            #
            # 1/1 note in 6/4 => 1x 1/1 note per bar
            # range(notes_per_bar) => [0] => 1 item in list, with item value == 0
            # note_index % 2 == 0 => 0 % 2 == 0 => True => note on @ time == 0
            notes_per_bar = (int(note_length.split('/')[1]) / denominator) * numerator
            note_ratio = (numerator / denominator) / (int(note_length.split('/')[0]) / int(note_length.split('/')[1]))
            if (note_ratio) != int(note_ratio):
                print(f"Skipping because note ratio was not a nice round number: {note_ratio}. I can't be arsed to cook up the logic for alterating 1/2 notes in 3/8 :B")
                continue
            print(f"Note length = {note_length} => {notes_per_bar} * {note_length} notes per bar")
            midi_tempo = mido.bpm2tempo(tempo, time_signature=(numerator, denominator))
            # 480 ticks per 1/4 note
            # 240 ticks per 1/8 note
            # 960 ticks per 1/2 note
            note_length_in_ticks = 480 * (4 / int(note_length.split('/')[1]))
            print(f"note length in ticks = {note_length_in_ticks}")
            note_length_in_seconds = mido.tick2second(note_length_in_ticks, 480, midi_tempo)
            print(f"note length in seconds = {note_length_in_seconds}")
            print(f"Generating MIDI sequence of {len([i for i in range(int(float(bars) * notes_per_bar))])} alternating {note_length} notes and rests over {bars} bars at {tempo} BPM in {time_signature}...")
            for midi_note_number in piano_midi_notes_all:
                print(f"MIDI note number = {midi_note_number}")
                print(f"MIDI note name = {number_to_note(midi_note_number)}")
                midi_sequence = []
                for note_index in range(int(float(bars) * notes_per_bar)):
                    if note_index % 2 == 0:
                        midi_sequence.append(Message("note_on", note=midi_note_number, time=(note_length_in_seconds * note_index)))
                        # 1/8 > ((3/8) / 2) = 0.125 > 0.1875 = False
                        # 0   1   2   3   4   5
                        # on  off on  off on  off
                    else:
                        midi_sequence.append(Message("note_off", note=midi_note_number, time=(note_length_in_seconds * note_index)))
                midi_sequence_duration = len(midi_sequence) * note_length_in_seconds
                print(f"MIDI sequence duration = {midi_sequence_duration} seconds")
                # 1/4 notes with ADSR_time_divisor == 2 will have attack/release/ times of 1/8 note, for example
                # Ergo it will take 1/8 note, or 1/2 of the 1/4 note, for the attack curve to max out,
                # and it will take another 1/8 note for the release time, so the 1/4 note's full audio length will be:
                # 1/8 attack + 1/8 hold and then decay to sustain level + 1/8 release, for a total length of 3/8 notes, 1/8 longer than 1/4 note.
                # 1/1 note in 6/4 = 1/4 attack + 3/4 hold + 1/2 release = 6/4 total time
                ADSR_time_divisor = 4
                attack_time = ''.join(['1/', str(int(note_length.split('/')[1]) * ADSR_time_divisor)])
                hold_time = attack_time
                decay_time = attack_time
                sustain_level = '-6.0 dB'
                release_time = attack_time
                decay_time = attack_time
                serum2.env_1_attack = attack_time
                serum2.env_1_hold = hold_time
                serum2.env_1_decay = decay_time
                serum2.env_1_sustain = sustain_level
                serum2.env_1_release = release_time
                serum2.env_1_atk_curve = 67.0
                serum2.env_1_dec_curve = 67.0
                serum2.env_1_rel_curve = 67.0
                print(f"ADSR => {serum2.env_1_attack} attack {serum2.env_1_hold} hold {serum2.env_1_decay} decay {serum2.env_1_sustain} sustain {serum2.env_1_release} release")
                serum2.a_unison = 2.0
                serum2.a_uni_detune = 0.25
                serum2.a_uni_blend = 75.0
                serum2.a_uni_width = 100.0
                print(f"Detune => {serum2.a_unison} voices {serum2.a_uni_detune} detune {serum2.a_uni_blend} blend {serum2.a_uni_width} width")
                print(f"Generating audio for {len([i for i in range(int(float(bars) * notes_per_bar))])} alternating {note_length} notes and rests over {bars} bars at {tempo} BPM in {time_signature}...")
                audio = serum2(midi_messages=midi_sequence, duration=midi_sequence_duration, sample_rate=sample_rate, num_channels=num_channels)
                folder_name = f"{preset_name}_{number_to_note(midi_note_number)}_{bars}bars"
                if not os.path.exists(os.path.join(os.getcwd(), 'renders', '2bars', folder_name)):
                    os.makedirs(os.path.join(os.getcwd(), 'renders', '2bars', folder_name), exist_ok=True)
                render_filename = f"{folder_name}_{tempo}BPM_alternating{int(note_length.split('/')[1])}notes_{numerator}n{denominator}d.wav"
                file_path = os.path.join(os.getcwd(), 'renders', '2bars', folder_name, render_filename)
                with pb.io.AudioFile(file_path, "w", sample_rate, num_channels=num_channels, bit_depth=32) as f:
                    f.write(audio)




>>> preset_name = 'Serum2_Triangle_randADSRdiv_-6dBsustain_67curve_180phase_0rand_randf1type_randf1hz_randf1res_mono'
for tempo in tempos:
...     for time_signature in time_signatures:
...         for note_length in note_lengths:
...             print(f"Tempo = {tempo}")
...             numerator, denominator = map(int, time_signature.split('/'))
...             print(f"Time Signature = {time_signature} = {numerator} * 1/{denominator} notes per bar")
...             # some note lengths may not make sense in certain time signatures
...             # if 1/4 note in 3/4, then 4 / 4 / 3, ergo 1 / 3, which is < 1
...             # if 1/1 note in 9/8, then 8 / 1 / 9, ergo 8/9, which is < 1
...             # if 1/1 note in 3/4, then 4 / 1 / 3, ergo 4 / 3, which is > 1, so skip it
...             if denominator / int(note_length.split('/')[1]) / numerator > 1:
...                 print(f"Skipping because {denominator} divided by {int(note_length.split('/')[1])} divided by {numerator} is > 1.0")
...                 continue
...             # fun fact:
...             # the correct way to write a note that lasts for a full 9/8 bar is to tie a dotted minim (dotted half-note) to a dotted crotchet (dotted quarter-note).
...             # 1.5 * 1/2 = 0.5 + 0.25 = dotted minim (dotted half-note)
...             # 1.5 * 1/4 = 0.25 + 0.125 = dotted crotchet (dotted quarter-note)
...             # Total = 1.125
...             # A "whole note" in 9/8 would be 8 * 8th notes.
...             #
...             # 1/1 note in 6/4 => 1x 1/1 note per bar
...             # range(notes_per_bar) => [0] => 1 item in list, with item value == 0
...             # note_index % 2 == 0 => 0 % 2 == 0 => True => note on @ time == 0
...             notes_per_bar = (int(note_length.split('/')[1]) / denominator) * numerator
...             note_ratio = (numerator / denominator) / (int(note_length.split('/')[0]) / int(note_length.split('/')[1]))
...             if (note_ratio) != int(note_ratio):
...                 print(f"Skipping because note ratio was not a nice round number: {note_ratio}. I can't be arsed to cook up the logic for alterating 1/2 notes in 3/8 :B")
...                 continue
...             print(f"Note length = {note_length} => {notes_per_bar} * {note_length} notes per bar")
...             midi_tempo = mido.bpm2tempo(tempo, time_signature=(numerator, denominator))
...             # 480 ticks per 1/4 note
...             # 240 ticks per 1/8 note
...             # 960 ticks per 1/2 note
...             note_length_in_ticks = 480 * (4 / int(note_length.split('/')[1]))
...             print(f"note length in ticks = {note_length_in_ticks}")
...             note_length_in_seconds = mido.tick2second(note_length_in_ticks, 480, midi_tempo)
...             print(f"note length in seconds = {note_length_in_seconds}")
...             print(f"Generating MIDI sequence of {len([i for i in range(int(float(bars) * notes_per_bar))])} alternating {note_length} notes and rests over {bars} bars at {tempo} BPM in {time_signature}...")
...             for midi_note_number in piano_midi_notes_all:
...                 print(f"MIDI note number = {midi_note_number}")
...                 print(f"MIDI note name = {number_to_note(midi_note_number)}")
...                 midi_sequence = []
...                 for note_index in range(int(float(bars) * notes_per_bar)):
...                     if note_index % 2 == 0:
...                         midi_sequence.append(Message("note_on", note=midi_note_number, time=(note_length_in_seconds * note_index)))
...                         # 1/8 > ((3/8) / 2) = 0.125 > 0.1875 = False
...                         # 0   1   2   3   4   5
...                         # on  off on  off on  off
...                     else:
...                         midi_sequence.append(Message("note_off", note=midi_note_number, time=(note_length_in_seconds * note_index)))
...                 midi_sequence_duration = len(midi_sequence) * note_length_in_seconds
...                 print(f"MIDI sequence duration = {midi_sequence_duration} seconds")
...                 # 1/4 notes with ADSR_time_divisor == 2 will have attack/release/ times of 1/8 note, for example
...                 # Ergo it will take 1/8 note, or 1/2 of the 1/4 note, for the attack curve to max out,
...                 # and it will take another 1/8 note for the release time, so the 1/4 note's full audio length will be:
...                 # 1/8 attack + 1/8 hold and then decay to sustain level + 1/8 release, for a total length of 3/8 notes, 1/8 longer than 1/4 note.
...                 # 1/1 note in 6/4 = 1/4 attack + 3/4 hold + 1/2 release = 6/4 total time
...                 ADSR_time_divisor = random.choice([2, 4, 8])
...                 attack_time = ''.join(['1/', str(int(note_length.split('/')[1]) * ADSR_time_divisor)])
...                 hold_time = attack_time
...                 decay_time = attack_time
...                 sustain_level = '-6.0 dB'
...                 release_time = attack_time
...                 decay_time = attack_time
...                 serum2.env_1_attack = attack_time
...                 serum2.env_1_hold = hold_time
...                 serum2.env_1_decay = decay_time
...                 serum2.env_1_sustain = sustain_level
...                 serum2.env_1_release = release_time
...                 serum2.env_1_atk_curve = 67.0
...                 serum2.env_1_dec_curve = 67.0
...                 serum2.env_1_rel_curve = 67.0
...                 print(f"ADSR => {serum2.env_1_attack} attack {serum2.env_1_hold} hold {serum2.env_1_decay} decay {serum2.env_1_sustain} sustain {serum2.env_1_release} release")
...                 serum2.a_unison = 1.0
...                 serum2.a_uni_detune = 0.25
...                 serum2.a_uni_blend = 75.0
...                 serum2.a_uni_width = 100.0
...                 print(f"Detune => {serum2.a_unison} voices {serum2.a_uni_detune} detune {serum2.a_uni_blend} blend {serum2.a_uni_width} width")
...                 serum2.filter_1_on = True
...                 serum2.filter_1_wet = 100.0
...                 serum2.filter_1_drive = 0.0
...                 serum2.filter_1_stereo = 50.0
...                 serum2.filter_1_var = 0.0
...                 serum2.filter_1_x = 0.0
...                 serum2.filter_1_y = 0.0
...                 serum2.filter_1_type = random.choice(serum2.filter_1_type.valid_values)
...                 serum2.filter_1_freq_hz = random.choice(serum2.filter_1_freq_hz.valid_values)
...                 serum2.filter_1_res = random.choice(serum2.filter_1_res.valid_values)
...                 print(f"Filter 1 => {serum2.filter_1_type} @ {serum2.filter_1_freq_hz} Hz {serum2.filter_1_res}% resonance")
...                 print(f"Generating audio for {len([i for i in range(int(float(bars) * notes_per_bar))])} alternating {note_length} notes and rests over {bars} bars at {tempo} BPM in {time_signature}...")
...                 audio = serum2(midi_messages=midi_sequence, duration=midi_sequence_duration, sample_rate=sample_rate, num_channels=num_channels)
...                 folder_name = f"{preset_name}_{number_to_note(midi_note_number)}_{bars}bars"
...                 if not os.path.exists(os.path.join('D:', 'renders', '2bars', folder_name)):
...                     os.makedirs(os.path.join('D:', 'renders', '2bars', folder_name), exist_ok=True)
...                 render_filename = f"{folder_name}_{tempo}BPM_alternating{int(note_length.split('/')[1])}notes_{numerator}n{denominator}d.wav"
...                 # Save JSON parameter states, i.e. preset/patch
...                 param_value_dict = {parameter_name: getattr(serum2, parameter_name) for parameter_name in serum2.parameters.keys()}
...                 # Convert pedalboard._pedalboard.WrappedBool to True/False
...                 param_value_dict = {k: (bool(v) if isinstance(v, WrappedBool) else v) for k, v in param_value_dict.items()}
...                 print(f"Saving JSON parameter dict: {render_filename}_params.json")
...                 with open(f"{render_filename}_params.json", 'w') as file:
...                     json.dump(param_value_dict, file, indent=4, ensure_ascii=True)
...                 # save raw state
...                 print(f"Saving raw plugin state binary file: {render_filename}_params.bin")
...                 with open(f"{render_filename}_params.bin", 'wb') as file:
...                     file.write(serum2.raw_state)
...                 print(f"Saving audio render @ {sample_rate} Hz and depth of 32-bit floating point across {num_channels} channels...")
...                 file_path = os.path.join('D:', 'renders', '2bars', folder_name, render_filename)
...                 with pb.io.AudioFile(file_path, "w", sample_rate, num_channels=num_channels, bit_depth=32) as f:
...                     f.write(audio)










import json
import os
import random
import re
import mido
import pedalboard as pb
from pprint import pprint
from mido import Message
from pedalboard._pedalboard import WrappedBool
from tqdm import tqdm

# A0 = 21
# C8 = 108
piano_midi_notes_all = range(21,109)

sample_rate = 48000

num_channels = 2

# 60 75 90 105 120 135 150 165 180 195
tempos = [tempo for tempo in range(60, 210, 15)]

minimal_tempos = [
    90,
    120,
    150,
    180
]

random_tempos = [
    random.randint(60, 89),
    random.randint(90, 119),
    random.randint(120, 149),
    random.randint(150, 179),
    random.randint(180, 200)
]

common_EDM_tempos = [
    100,
    110,
    120,
    128,
    135,
    140,
    150,
    175
]

time_signatures = [
    '2/2',
    '3/2',
    '4/2',
    '2/4',
    '3/4',
    '4/4',
    '5/4',
    '6/4',
    '7/4',
    '8/4',
    '3/8',
    '4/8',
    '5/8',
    '6/8',
    '7/8',
    '8/8',
    '9/8',
    '12/8'
]

minimal_time_signatures = [
    '3/4',
    '4/4',
    '5/4',
    '6/8',
    '7/8',
    '8/8',
    '9/8',
    '12/8'
]

note_lengths = [
    '1/32',
    '1/16',
    '1/8',
    '1/4'
]

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
octaves = list(range(11))
notes_per_octave = len(note_names)


def number_to_note(number: int) -> tuple:
    note_name = note_names[number % notes_per_octave]
    note_octave = (number // notes_per_octave) - 1
    # F#0, C3, G9, etc.
    return ''.join([note_name, str(note_octave)])


def note_to_number(note: str) -> int:
    octave = int(note[-1]) + 1
    note_number = note_names.index(note[0:-1])
    note_number += (notes_per_octave * octave)
    return note_number


serum2 = pb.load_plugin('Serum2.vst3', plugin_name='Serum 2') 

bars = 2

waveform = 'SawUp'


for tempo in tqdm(random_tempos, unit='tempo'):
    for time_signature in tqdm(minimal_time_signatures, unit='time_signature'):
        for note_length in tqdm(note_lengths, unit='note_length'):
            print(f"Tempo = {tempo}")
            numerator, denominator = map(int, time_signature.split('/'))
            print(f"Time Signature = {time_signature} = {numerator} * 1/{denominator} notes per bar")
            # some note lengths may not make sense in certain time signatures
            # if 1/4 note in 3/4, then 4 / 4 / 3, ergo 1 / 3, which is < 1
            # if 1/1 note in 9/8, then 8 / 1 / 9, ergo 8/9, which is < 1
            # if 1/1 note in 3/4, then 4 / 1 / 3, ergo 4 / 3, which is > 1, so skip it
            if denominator / int(note_length.split('/')[1]) / numerator > 1:
                print(f"Skipping because {denominator} divided by {int(note_length.split('/')[1])} divided by {numerator} is > 1.0")
                continue
            # fun fact:
            # the correct way to write a note that lasts for a full 9/8 bar is to tie a dotted minim (dotted half-note) to a dotted crotchet (dotted quarter-note).
            # 1.5 * 1/2 = 0.5 + 0.25 = dotted minim (dotted half-note)
            # 1.5 * 1/4 = 0.25 + 0.125 = dotted crotchet (dotted quarter-note)
            # Total = 1.125
            # A "whole note" in 9/8 would be 8 * 8th notes.
            #
            # 1/1 note in 6/4 => 1x 1/1 note per bar
            # range(notes_per_bar) => [0] => 1 item in list, with item value == 0
            # note_index % 2 == 0 => 0 % 2 == 0 => True => note on @ time == 0
            notes_per_bar = (int(note_length.split('/')[1]) / denominator) * numerator
            note_ratio = (numerator / denominator) / (int(note_length.split('/')[0]) / int(note_length.split('/')[1]))
            if (note_ratio * float(bars)) != int(note_ratio * float(bars)):
                print(f"Skipping because note ratio was not a nice round number: {note_ratio}. I can't be arsed to cook up the logic for alterating 1/2 notes in 1x bar of 3/8 :B")
                continue
            print(f"Note length = {note_length} => {notes_per_bar} * {note_length} notes per bar")
            midi_tempo = mido.bpm2tempo(tempo, time_signature=(numerator, denominator))
            # 480 ticks per 1/4 note
            # 240 ticks per 1/8 note
            # 960 ticks per 1/2 note
            note_length_in_ticks = 480 * (4 / int(note_length.split('/')[1]))
            print(f"note length in ticks = {note_length_in_ticks}")
            note_length_in_seconds = mido.tick2second(note_length_in_ticks, 480, midi_tempo)
            print(f"note length in seconds = {note_length_in_seconds}")
            total_notes_per_sequence = int(float(bars) * notes_per_bar)
            print(f"Generating MIDI sequence of {len([i for i in range(total_notes_per_sequence)])} alternating {note_length} notes and rests over {bars} bars at {tempo} BPM in {time_signature}...")
            for midi_note_number in random.sample(piano_midi_notes_all, k=12):
                print(f"MIDI note number = {midi_note_number}")
                print(f"MIDI note name = {number_to_note(midi_note_number)}")
                midi_sequence = []
                for note_index in range(total_notes_per_sequence):
                    if note_index % 2 == 0:
                        midi_sequence.append(Message("note_on", note=midi_note_number, time=(note_length_in_seconds * note_index)))
                        # 1/8 > ((3/8) / 2) = 0.125 > 0.1875 = False
                        # 0   1   2   3   4   5
                        # on  off on  off on  off
                    else:
                        midi_sequence.append(Message("note_off", note=midi_note_number, time=(note_length_in_seconds * note_index)))
                midi_sequence_duration = len(midi_sequence) * note_length_in_seconds
                print(f"MIDI sequence duration = {midi_sequence_duration} seconds")
                # 1/4 notes with ADSR_time_divisor == 2 will have attack/release/ times of 1/8 note, for example
                # Ergo it will take 1/8 note, or 1/2 of the 1/4 note, for the attack curve to max out,
                # and it will take another 1/8 note for the release time, so the 1/4 note's full audio length will be:
                # 1/8 attack + 1/8 hold and then decay to sustain level + 1/8 release, for a total length of 3/8 notes, 1/8 longer than 1/4 note.
                # 1/1 note in 6/4 = 1/4 attack + 3/4 hold + 1/2 release = 6/4 total time
                attack_time = ''.join(['1/', str(int(note_length.split('/')[1]) * random.choice([2, 4, 8]))])
                hold_time = ''.join(['1/', str(int(note_length.split('/')[1]) * random.choice([2, 4, 8]))])
                decay_time = ''.join(['1/', str(int(note_length.split('/')[1]) * random.choice([2, 4, 8]))])
                sustain_level = f"{random.choice([-10.0, -6.0, -5.0, -3.0, 0.0])} dB"
                release_time = ''.join(['1/', str(int(note_length.split('/')[1]) * random.choice([2, 4, 8]))])
                serum2.env_1_attack = attack_time
                serum2.env_1_hold = hold_time
                serum2.env_1_decay = decay_time
                serum2.env_1_sustain = sustain_level
                serum2.env_1_release = release_time
                serum2.env_1_atk_curve = random.choice(serum2.env_1_atk_curve.valid_values)
                serum2.env_1_dec_curve = random.choice(serum2.env_1_dec_curve.valid_values)
                serum2.env_1_rel_curve = random.choice(serum2.env_1_rel_curve.valid_values)
                print(f"ADSR => {serum2.env_1_attack} attack {serum2.env_1_hold} hold {serum2.env_1_decay} decay {serum2.env_1_sustain} sustain {serum2.env_1_release} release")
                serum2.a_unison = random.choice(serum2.a_unison.valid_values) # 1.0
                serum2.a_uni_detune = random.choice(serum2.a_uni_detune.valid_values) # 0.25
                serum2.a_uni_blend = random.choice(serum2.a_uni_blend.valid_values) # 75.0
                serum2.a_uni_width = random.choice(serum2.a_uni_width.valid_values) # 100.0
                print(f"Detune => {serum2.a_unison} voices {serum2.a_uni_detune} detune {serum2.a_uni_blend} blend {serum2.a_uni_width} width")
                serum2.filter_1_on = True
                serum2.filter_1_wet = random.choice(serum2.filter_1_wet.valid_values) # 100.0
                serum2.filter_1_drive = random.choice(serum2.filter_1_drive.valid_values) # 0.0
                serum2.filter_1_stereo = random.choice(serum2.filter_1_stereo.valid_values) # 50.0
                serum2.filter_1_var = random.choice(serum2.filter_1_var.valid_values) # 0.0
                serum2.filter_1_x = 0.0
                serum2.filter_1_y = 0.0
                serum2.filter_1_type = random.choice(serum2.filter_1_type.valid_values)
                serum2.filter_1_freq_hz = random.choice(serum2.filter_1_freq_hz.valid_values)
                serum2.filter_1_res = random.choice(serum2.filter_1_res.valid_values)
                print(f"Filter 1 => {serum2.filter_1_type} @ {serum2.filter_1_freq_hz} Hz {serum2.filter_1_res}% resonance")
                print(f"Generating audio for {len([i for i in range(int(float(bars) * notes_per_bar))])} alternating {note_length} notes and rests over {bars} bars at {tempo} BPM in {time_signature}...")
                audio = serum2(midi_messages=midi_sequence, duration=midi_sequence_duration, sample_rate=sample_rate, num_channels=num_channels)
                esc_sus = serum2.env_1_sustain.split(' ')[0]
                esc_atk = serum2.env_1_attack.split('/')[1]
                esc_hold = serum2.env_1_hold.split('/')[1]
                esc_dec = serum2.env_1_decay.split('/')[1]
                esc_rel = serum2.env_1_release.split('/')[1]
                esc_f1type = re.sub(r'[^a-zA-Z0-9]', '', serum2.filter_1_type)
                preset_name = f"Serum2_{waveform}_{esc_atk}atk_{esc_hold}hold_{esc_dec}dec_{esc_sus}dBsus_{esc_rel}rel_{serum2.env_1_atk_curve}acurv_{serum2.env_1_dec_curve}dcurv_{serum2.env_1_rel_curve}rcurv_{esc_f1type}f1type_{serum2.filter_1_freq_hz}f1hz_{serum2.filter_1_res}f1res_180phase_0rand_mono"
                folder_name = f"{preset_name}_{number_to_note(midi_note_number)}_{bars}bars"
                #if not os.path.exists(os.path.join('D:', 'renders', '2bars', folder_name)):
                #    os.makedirs(os.path.join('D:', 'renders', '2bars', folder_name), exist_ok=True)
                render_filename = f"{folder_name}_{tempo}BPM_alternating{int(note_length.split('/')[1])}notes_{numerator}n{denominator}d.wav"
                # Save JSON parameter states, i.e. preset/patch
                param_value_dict = {parameter_name: getattr(serum2, parameter_name) for parameter_name in serum2.parameters.keys()}
                # Convert pedalboard._pedalboard.WrappedBool to True/False
                param_value_dict = {k: (bool(v) if isinstance(v, WrappedBool) else v) for k, v in param_value_dict.items()}
                print(f"Saving JSON parameter dict: {render_filename[0:-4]}_params.json")
                with open(os.path.join('D:', 'renders', '2bars', f"{render_filename[0:-4]}_params.json"), 'w') as file:
                    json.dump(param_value_dict, file, indent=4, ensure_ascii=True)
                # save raw state
                print(f"Saving raw plugin state binary file: {render_filename[0:-4]}_params.bin")
                with open(os.path.join('D:', 'renders', '2bars', f"{render_filename[0:-4]}_params.bin"), 'wb') as file:
                    file.write(serum2.raw_state)
                print(f"Saving audio render @ {sample_rate} Hz and depth of 32-bit floating point across {num_channels} channels...")
                file_path = os.path.join('D:', 'renders', '2bars', render_filename)
                with pb.io.AudioFile(file_path, "w", sample_rate, num_channels=num_channels, bit_depth=32) as f:
                    f.write(audio)






talunolxv2 = pb.load_plugin('C:/Program Files/Common Files/VST3/TAL-U-NO-LX-V2.vst3/Contents/x86_64-win/TAL-U-NO-LX-V2.vst3')

for tempo in tqdm(random_tempos, unit='tempo'):
    for time_signature in tqdm(minimal_time_signatures, unit='time_signature'):
        for note_length in tqdm(note_lengths, unit='note_length'):
            print(f"Tempo = {tempo}")
            numerator, denominator = map(int, time_signature.split('/'))
            print(f"Time Signature = {time_signature} = {numerator} * 1/{denominator} notes per bar")
            # some note lengths may not make sense in certain time signatures
            # if 1/4 note in 3/4, then 4 / 4 / 3, ergo 1 / 3, which is < 1
            # if 1/1 note in 9/8, then 8 / 1 / 9, ergo 8/9, which is < 1
            # if 1/1 note in 3/4, then 4 / 1 / 3, ergo 4 / 3, which is > 1, so skip it
            if denominator / int(note_length.split('/')[1]) / numerator > 1:
                print(f"Skipping because {denominator} divided by {int(note_length.split('/')[1])} divided by {numerator} is > 1.0")
                continue
            # fun fact:
            # the correct way to write a note that lasts for a full 9/8 bar is to tie a dotted minim (dotted half-note) to a dotted crotchet (dotted quarter-note).
            # 1.5 * 1/2 = 0.5 + 0.25 = dotted minim (dotted half-note)
            # 1.5 * 1/4 = 0.25 + 0.125 = dotted crotchet (dotted quarter-note)
            # Total = 1.125
            # A "whole note" in 9/8 would be 8 * 8th notes.
            #
            # 1/1 note in 6/4 => 1x 1/1 note per bar
            # range(notes_per_bar) => [0] => 1 item in list, with item value == 0
            # note_index % 2 == 0 => 0 % 2 == 0 => True => note on @ time == 0
            notes_per_bar = (int(note_length.split('/')[1]) / denominator) * numerator
            note_ratio = (numerator / denominator) / (int(note_length.split('/')[0]) / int(note_length.split('/')[1]))
            if (note_ratio * float(bars)) != int(note_ratio * float(bars)):
                print(f"Skipping because note ratio was not a nice round number: {note_ratio}. I can't be arsed to cook up the logic for alterating 1/2 notes in 1x bar of 3/8 :B")
                continue
            print(f"Note length = {note_length} => {notes_per_bar} * {note_length} notes per bar")
            midi_tempo = mido.bpm2tempo(tempo, time_signature=(numerator, denominator))
            # 480 ticks per 1/4 note
            # 240 ticks per 1/8 note
            # 960 ticks per 1/2 note
            note_length_in_ticks = 480 * (4 / int(note_length.split('/')[1]))
            print(f"note length in ticks = {note_length_in_ticks}")
            note_length_in_seconds = mido.tick2second(note_length_in_ticks, 480, midi_tempo)
            print(f"note length in seconds = {note_length_in_seconds}")
            total_notes_per_sequence = int(float(bars) * notes_per_bar)
            print(f"Generating MIDI sequence of {len([i for i in range(total_notes_per_sequence)])} alternating {note_length} notes and rests over {bars} bars at {tempo} BPM in {time_signature}...")
            for midi_note_number in random.sample(piano_midi_notes_all, k=12):
                print(f"MIDI note number = {midi_note_number}")
                print(f"MIDI note name = {number_to_note(midi_note_number)}")
                midi_sequence = []
                for note_index in range(total_notes_per_sequence):
                    if note_index % 2 == 0:
                        midi_sequence.append(Message("note_on", note=midi_note_number, time=(note_length_in_seconds * note_index)))
                        # 1/8 > ((3/8) / 2) = 0.125 > 0.1875 = False
                        # 0   1   2   3   4   5
                        # on  off on  off on  off
                    else:
                        midi_sequence.append(Message("note_off", note=midi_note_number, time=(note_length_in_seconds * note_index)))
                midi_sequence_duration = len(midi_sequence) * note_length_in_seconds
                print(f"MIDI sequence duration = {midi_sequence_duration} seconds")
                print(f"Generating audio for {len([i for i in range(int(float(bars) * notes_per_bar))])} alternating {note_length} notes and rests over {bars} bars at {tempo} BPM in {time_signature}...")
                audio = talunolxv2(midi_messages=midi_sequence, duration=midi_sequence_duration, sample_rate=sample_rate, num_channels=num_channels)
                preset_name = f"talunolxv2_BASPulseBass1"
                folder_name = f"{preset_name}_{number_to_note(midi_note_number)}_{bars}bars"
                #if not os.path.exists(os.path.join('D:', 'renders', '2bars', folder_name)):
                #    os.makedirs(os.path.join('D:', 'renders', '2bars', folder_name), exist_ok=True)
                render_filename = f"{folder_name}_{tempo}BPM_alternating{int(note_length.split('/')[1])}notes_{numerator}n{denominator}d.wav"
                # Save JSON parameter states, i.e. preset/patch
                param_value_dict = {parameter_name: getattr(talunolxv2, parameter_name) for parameter_name in talunolxv2.parameters.keys()}
                # Convert pedalboard._pedalboard.WrappedBool to True/False
                param_value_dict = {k: (bool(v) if isinstance(v, WrappedBool) else v) for k, v in param_value_dict.items()}
                print(f"Saving JSON parameter dict: {render_filename[0:-4]}_params.json")
                with open(os.path.join('D:', 'renders', '2bars', f"{render_filename[0:-4]}_params.json"), 'w') as file:
                    json.dump(param_value_dict, file, indent=4, ensure_ascii=True)
                # save raw state
                print(f"Saving raw plugin state binary file: {render_filename[0:-4]}_params.bin")
                with open(os.path.join('D:', 'renders', '2bars', f"{render_filename[0:-4]}_params.bin"), 'wb') as file:
                    file.write(talunolxv2.raw_state)
                print(f"Saving audio render @ {sample_rate} Hz and depth of 32-bit floating point across {num_channels} channels...")
                file_path = os.path.join('D:', 'renders', '2bars', render_filename)
                with pb.io.AudioFile(file_path, "w", sample_rate, num_channels=num_channels, bit_depth=32) as f:
                    f.write(audio)






>>> for tempo in tqdm(random_tempos, unit='tempo'):
...     for time_signature in tqdm(minimal_time_signatures, unit='time_signature'):
...         for note_length in tqdm(note_lengths, unit='note_length'):
...             print(f"Tempo = {tempo}")
...             numerator, denominator = map(int, time_signature.split('/'))
...             print(f"Time Signature = {time_signature} = {numerator} * 1/{denominator} notes per bar")
...             # some note lengths may not make sense in certain time signatures
...             # if 1/4 note in 3/4, then 4 / 4 / 3, ergo 1 / 3, which is < 1
...             # if 1/1 note in 9/8, then 8 / 1 / 9, ergo 8/9, which is < 1
...             # if 1/1 note in 3/4, then 4 / 1 / 3, ergo 4 / 3, which is > 1, so skip it
...             if denominator / int(note_length.split('/')[1]) / numerator > 1:
...                 print(f"Skipping because {denominator} divided by {int(note_length.split('/')[1])} divided by {numerator} is > 1.0")
...                 continue
...             # fun fact:
...             # the correct way to write a note that lasts for a full 9/8 bar is to tie a dotted minim (dotted half-note) to a dotted crotchet (dotted quarter-note).
...             # 1.5 * 1/2 = 0.5 + 0.25 = dotted minim (dotted half-note)
...             # 1.5 * 1/4 = 0.25 + 0.125 = dotted crotchet (dotted quarter-note)
...             # Total = 1.125
...             # A "whole note" in 9/8 would be 8 * 8th notes.
...             #
...             # 1/1 note in 6/4 => 1x 1/1 note per bar
...             # range(notes_per_bar) => [0] => 1 item in list, with item value == 0
...             # note_index % 2 == 0 => 0 % 2 == 0 => True => note on @ time == 0
...             notes_per_bar = (int(note_length.split('/')[1]) / denominator) * numerator
...             note_ratio = (numerator / denominator) / (int(note_length.split('/')[0]) / int(note_length.split('/')[1]))
...             if (note_ratio * float(bars)) != int(note_ratio * float(bars)):
...                 print(f"Skipping because note ratio was not a nice round number: {note_ratio}. I can't be arsed to cook up the logic for alterating 1/2 notes in 1x bar of 3/8 :B")
...                 continue
...             print(f"Note length = {note_length} => {notes_per_bar} * {note_length} notes per bar")
...             midi_tempo = mido.bpm2tempo(tempo, time_signature=(numerator, denominator))
...             # 480 ticks per 1/4 note
...             # 240 ticks per 1/8 note
...             # 960 ticks per 1/2 note
...             note_length_in_ticks = 480 * (4 / int(note_length.split('/')[1]))
...             print(f"note length in ticks = {note_length_in_ticks}")
...             note_length_in_seconds = mido.tick2second(note_length_in_ticks, 480, midi_tempo)
...             print(f"note length in seconds = {note_length_in_seconds}")
...             total_notes_per_sequence = int(float(bars) * notes_per_bar)
...             print(f"Generating MIDI sequence of {len([i for i in range(total_notes_per_sequence)])} alternating {note_length} notes and rests over {bars} bars at {tempo} BPM in {time_signature}...")
...             for midi_note_number in random.sample(piano_midi_notes_all, k=12):
...                 print(f"MIDI note number = {midi_note_number}")
...                 print(f"MIDI note name = {number_to_note(midi_note_number)}")
...                 midi_sequence = []
...                 for note_index in range(total_notes_per_sequence):
...                     if note_index % 2 == 0:
...                         midi_sequence.append(Message("note_on", note=midi_note_number, time=(note_length_in_seconds * note_index)))
...                         # 1/8 > ((3/8) / 2) = 0.125 > 0.1875 = False
...                         # 0   1   2   3   4   5
...                         # on  off on  off on  off
...                     else:
...                         midi_sequence.append(Message("note_off", note=midi_note_number, time=(note_length_in_seconds * note_index)))
...                 midi_sequence_duration = len(midi_sequence) * note_length_in_seconds
...                 print(f"MIDI sequence duration = {midi_sequence_duration} seconds")
...                 print(f"Generating audio for {len([i for i in range(int(float(bars) * notes_per_bar))])} alternating {note_length} notes and rests over {bars} bars at {tempo} BPM in {time_signature}...")
...                 audio = talj8(midi_messages=midi_sequence, duration=midi_sequence_duration, sample_rate=sample_rate, num_channels=num_channels)
...                 preset_name = f"talj8_UnisonSawLead"
...                 folder_name = f"{preset_name}_{number_to_note(midi_note_number)}_{bars}bars"
...                 #if not os.path.exists(os.path.join('D:', 'renders', '2bars', folder_name)):
...                 #    os.makedirs(os.path.join('D:', 'renders', '2bars', folder_name), exist_ok=True)
...                 render_filename = f"{folder_name}_{tempo}BPM_alternating{int(note_length.split('/')[1])}notes_{numerator}n{denominator}d.wav"
...                 # Save JSON parameter states, i.e. preset/patch
...                 param_value_dict = {parameter_name: getattr(talj8, parameter_name) for parameter_name in talj8.parameters.keys()}
...                 # Convert pedalboard._pedalboard.WrappedBool to True/False
...                 param_value_dict = {k: (bool(v) if isinstance(v, WrappedBool) else v) for k, v in param_value_dict.items()}
...                 print(f"Saving JSON parameter dict: {render_filename[0:-4]}_params.json")
...                 with open(os.path.join('D:', 'renders', '2bars', f"{render_filename[0:-4]}_params.json"), 'w') as file:
...                     json.dump(param_value_dict, file, indent=4, ensure_ascii=True)
...                 # save raw state
...                 print(f"Saving raw plugin state binary file: {render_filename[0:-4]}_params.bin")
...                 with open(os.path.join('D:', 'renders', '2bars', f"{render_filename[0:-4]}_params.bin"), 'wb') as file:
...                     file.write(talj8.raw_state)
...                 print(f"Saving audio render @ {sample_rate} Hz and depth of 32-bit floating point across {num_channels} channels...")
...                 file_path = os.path.join('D:', 'renders', '2bars', render_filename)
...                 with pb.io.AudioFile(file_path, "w", sample_rate, num_channels=num_channels, bit_depth=32) as f:
...                     f.write(audio)





>>> for tempo in tqdm(random_tempos, unit='tempo'):
...     for time_signature in tqdm(minimal_time_signatures, unit='time_signature'):
...         for note_length in tqdm(note_lengths, unit='note_length'):
...             print(f"Tempo = {tempo}")
...             numerator, denominator = map(int, time_signature.split('/'))
...             print(f"Time Signature = {time_signature} = {numerator} * 1/{denominator} notes per bar")
...             # some note lengths may not make sense in certain time signatures
...             # if 1/4 note in 3/4, then 4 / 4 / 3, ergo 1 / 3, which is < 1
...             # if 1/1 note in 9/8, then 8 / 1 / 9, ergo 8/9, which is < 1
...             # if 1/1 note in 3/4, then 4 / 1 / 3, ergo 4 / 3, which is > 1, so skip it
...             if denominator / int(note_length.split('/')[1]) / numerator > 1:
...                 print(f"Skipping because {denominator} divided by {int(note_length.split('/')[1])} divided by {numerator} is > 1.0")
...                 continue
...             # fun fact:
...             # the correct way to write a note that lasts for a full 9/8 bar is to tie a dotted minim (dotted half-note) to a dotted crotchet (dotted quarter-note).
...             # 1.5 * 1/2 = 0.5 + 0.25 = dotted minim (dotted half-note)
...             # 1.5 * 1/4 = 0.25 + 0.125 = dotted crotchet (dotted quarter-note)
...             # Total = 1.125
...             # A "whole note" in 9/8 would be 8 * 8th notes.
...             #
...             # 1/1 note in 6/4 => 1x 1/1 note per bar
...             # range(notes_per_bar) => [0] => 1 item in list, with item value == 0
...             # note_index % 2 == 0 => 0 % 2 == 0 => True => note on @ time == 0
...             notes_per_bar = (int(note_length.split('/')[1]) / denominator) * numerator
...             note_ratio = (numerator / denominator) / (int(note_length.split('/')[0]) / int(note_length.split('/')[1]))
...             if (note_ratio * float(bars)) != int(note_ratio * float(bars)):
...                 print(f"Skipping because note ratio was not a nice round number: {note_ratio}. I can't be arsed to cook up the logic for alterating 1/2 notes in 1x bar of 3/8 :B")
...                 continue
...             print(f"Note length = {note_length} => {notes_per_bar} * {note_length} notes per bar")
...             midi_tempo = mido.bpm2tempo(tempo, time_signature=(numerator, denominator))
...             # 480 ticks per 1/4 note
...             # 240 ticks per 1/8 note
...             # 960 ticks per 1/2 note
...             note_length_in_ticks = 480 * (4 / int(note_length.split('/')[1]))
...             print(f"note length in ticks = {note_length_in_ticks}")
...             note_length_in_seconds = mido.tick2second(note_length_in_ticks, 480, midi_tempo)
...             print(f"note length in seconds = {note_length_in_seconds}")
...             total_notes_per_sequence = int(float(bars) * notes_per_bar)
...             print(f"Generating MIDI sequence of {len([i for i in range(total_notes_per_sequence)])} alternating {note_length} notes and rests over {bars} bars at {tempo} BPM in {time_signature}...")
...             for midi_note_number in random.sample(piano_midi_notes_all, k=12):
...                 print(f"MIDI note number = {midi_note_number}")
...                 print(f"MIDI note name = {number_to_note(midi_note_number)}")
...                 midi_sequence = []
...                 for note_index in range(total_notes_per_sequence):
...                     if note_index % 2 == 0:
...                         midi_sequence.append(Message("note_on", note=midi_note_number, time=(note_length_in_seconds * note_index)))
...                         # 1/8 > ((3/8) / 2) = 0.125 > 0.1875 = False
...                         # 0   1   2   3   4   5
...                         # on  off on  off on  off
...                     else:
...                         midi_sequence.append(Message("note_off", note=midi_note_number, time=(note_length_in_seconds * note_index)))
...                 midi_sequence_duration = len(midi_sequence) * note_length_in_seconds
...                 print(f"MIDI sequence duration = {midi_sequence_duration} seconds")
...                 print(f"Generating audio for {len([i for i in range(int(float(bars) * notes_per_bar))])} alternating {note_length} notes and rests over {bars} bars at {tempo} BPM in {time_signature}...")
...                 audio = talunolxv2(midi_messages=midi_sequence, duration=midi_sequence_duration, sample_rate=sample_rate, num_channels=num_channels)
...                 preset_name = f"talunolxv2_BASAcidBass1FMR"
...                 folder_name = f"{preset_name}_{number_to_note(midi_note_number)}_{bars}bars"
...                 if not os.path.exists(os.path.join('P:', 'renders', '2bars', preset_name)):
...                     os.makedirs(os.path.join('P:', 'renders', '2bars', preset_name), exist_ok=True)
...                 render_filename = f"{folder_name}_{tempo}BPM_alternating{int(note_length.split('/')[1])}notes_{numerator}n{denominator}d.wav"
...                 # Save JSON parameter states, i.e. preset/patch
...                 param_value_dict = {parameter_name: getattr(talunolxv2, parameter_name) for parameter_name in talunolxv2.parameters.keys()}
...                 # Convert pedalboard._pedalboard.WrappedBool to True/False
...                 param_value_dict = {k: (bool(v) if isinstance(v, WrappedBool) else v) for k, v in param_value_dict.items()}
...                 print(f"Saving JSON parameter dict: {render_filename[0:-4]}_params.json")
...                 with open(os.path.join('P:', 'renders', '2bars', preset_name, f"{render_filename[0:-4]}_params.json"), 'w') as file:
...                     json.dump(param_value_dict, file, indent=4, ensure_ascii=True)
...                 # save raw state
...                 print(f"Saving raw plugin state binary file: {render_filename[0:-4]}_params.bin")
...                 with open(os.path.join('P:', 'renders', '2bars', preset_name, f"{render_filename[0:-4]}_params.bin"), 'wb') as file:
...                     file.write(talunolxv2.raw_state)
...                 print(f"Saving audio render @ {sample_rate} Hz and depth of 32-bit floating point across {num_channels} channels...")
...                 file_path = os.path.join('P:', 'renders', '2bars', preset_name, render_filename)
...                 with pb.io.AudioFile(file_path, "w", sample_rate, num_channels=num_channels, bit_depth=32) as f:
...                     f.write(audio)






import json
import os
import random
import re
import time

import mido
import pedalboard as pb

from pprint import pprint
from threading import Event, Timer

from mido import Message
from pedalboard._pedalboard import WrappedBool
from tqdm import tqdm

# A0 = 21
# C8 = 108
piano_midi_notes_all = range(21,109)

sample_rate = 48000

num_channels = 2

# 60 75 90 105 120 135 150 165 180 195
tempos = [tempo for tempo in range(60, 210, 15)]

minimal_tempos = [
    90,
    120,
    150,
    180
]

random_tempos = [
    random.randint(60, 89),
    random.randint(90, 119),
    random.randint(120, 149),
    random.randint(150, 179),
    random.randint(180, 200)
]

common_EDM_tempos = [
    100,
    110,
    120,
    128,
    135,
    140,
    150,
    175
]

time_signatures = [
    '2/2',
    '3/2',
    '4/2',
    '2/4',
    '3/4',
    '4/4',
    '5/4',
    '6/4',
    '7/4',
    '8/4',
    '3/8',
    '4/8',
    '5/8',
    '6/8',
    '7/8',
    '8/8',
    '9/8',
    '12/8'
]

minimal_time_signatures = [
    '3/4',
    '4/4',
    '5/4',
    '6/8',
    '7/8',
    '8/8',
    '9/8',
    '12/8'
]

note_lengths = [
    '1/32',
    '1/16',
    '1/8',
    '1/4'
]

note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
octaves = list(range(11))
notes_per_octave = len(note_names)


def number_to_note(number: int) -> tuple:
    note_name = note_names[number % notes_per_octave]
    note_octave = (number // notes_per_octave) - 1
    # F#0, C3, G9, etc.
    return ''.join([note_name, str(note_octave)])


def note_to_number(note: str) -> int:
    octave = int(note[-1]) + 1
    note_number = note_names.index(note[0:-1])
    note_number += (notes_per_octave * octave)
    return note_number


def random_tempos():
    random_tempos = [
        random.randint(60, 89),
        random.randint(90, 119),
        random.randint(120, 149),
        random.randint(150, 179),
        random.randint(180, 200)
    ]
    return random_tempos


def dummy_thread():
    time.sleep(1)
    print("dummy thread called, slept 1 second, close_window_event.set()")
    close_window_event.set()


bars = 2

# Load VST3 plugins
#serum2 = pb.load_plugin('Serum2.vst3', plugin_name='Serum 2')
talunolxv2 = pb.load_plugin('C:/Program Files/Common Files/VST3/TAL-U-NO-LX-V2.vst3/Contents/x86_64-win/TAL-U-NO-LX-V2.vst3')

close_window_event = Event()

# Change presets/patches
talunolxv2(midi_messages=[Message('program_change', program=5, time=0)], duration=1, sample_rate=48000, num_channels=2)

# Open the plugin GUI for 16 seconds to refresh it
t = Timer(15, dummy_thread)
t.start()
talunolxv2.show_editor(close_window_event)

# Reset the event status
close_window_event.clear()

# Get preset name from raw binary preset data
talunolxv2_preset_name = re.search(r'programname="([^"]+)" ', str(talunolxv2.preset_data))[1]
talunolxv2_preset_name = re.sub(r'\W', '', talunolxv2_preset_name)
#serum2_preset_name = re.search(r'"presetName":"([^"]+)"', str(serum2.preset_data))[1]

diva = pb.load_plugin('C:/Program Files/Common Files/VST3/Diva(x64).vst3')
diva_preset_name = re.search(r'#pgm=([^\.]+)' , str(diva.preset_data))[1]

for tempo in tqdm(random_tempos, unit='tempo'):
    for time_signature in tqdm(minimal_time_signatures, unit='time_signature'):
        for note_length in tqdm(note_lengths, unit='note_length'):
            print(f"Tempo = {tempo}")
            numerator, denominator = map(int, time_signature.split('/'))
            print(f"Time Signature = {time_signature} = {numerator} * 1/{denominator} notes per bar")
            # some note lengths may not make sense in certain time signatures
            # if 1/4 note in 3/4, then 4 / 4 / 3, ergo 1 / 3, which is < 1
            # if 1/1 note in 9/8, then 8 / 1 / 9, ergo 8/9, which is < 1
            # if 1/1 note in 3/4, then 4 / 1 / 3, ergo 4 / 3, which is > 1, so skip it
            if denominator / int(note_length.split('/')[1]) / numerator > 1:
                print(f"Skipping because {denominator} divided by {int(note_length.split('/')[1])} divided by {numerator} is > 1.0")
                continue
            # fun fact:
            # the correct way to write a note that lasts for a full 9/8 bar is to tie a dotted minim (dotted half-note) to a dotted crotchet (dotted quarter-note).
            # 1.5 * 1/2 = 0.5 + 0.25 = dotted minim (dotted half-note)
            # 1.5 * 1/4 = 0.25 + 0.125 = dotted crotchet (dotted quarter-note)
            # Total = 1.125
            # A "whole note" in 9/8 would be 8 * 8th notes.
            #
            # 1/1 note in 6/4 => 1x 1/1 note per bar
            # range(notes_per_bar) => [0] => 1 item in list, with item value == 0
            # note_index % 2 == 0 => 0 % 2 == 0 => True => note on @ time == 0
            notes_per_bar = (int(note_length.split('/')[1]) / denominator) * numerator
            note_ratio = (numerator / denominator) / (int(note_length.split('/')[0]) / int(note_length.split('/')[1]))
            if (note_ratio * float(bars)) != int(note_ratio * float(bars)):
                print(f"Skipping because note ratio was not a nice round number: {note_ratio}. I can't be arsed to cook up the logic for alterating 1/2 notes in 1x bar of 3/8 :B")
                continue
            print(f"Note length = {note_length} => {notes_per_bar} * {note_length} notes per bar")
            midi_tempo = mido.bpm2tempo(tempo, time_signature=(numerator, denominator))
            # 480 ticks per 1/4 note
            # 240 ticks per 1/8 note
            # 960 ticks per 1/2 note
            note_length_in_ticks = 480 * (4 / int(note_length.split('/')[1]))
            print(f"note length in ticks = {note_length_in_ticks}")
            note_length_in_seconds = mido.tick2second(note_length_in_ticks, 480, midi_tempo)
            print(f"note length in seconds = {note_length_in_seconds}")
            total_notes_per_sequence = int(float(bars) * notes_per_bar)
            print(f"Generating MIDI sequence of {len([i for i in range(total_notes_per_sequence)])} alternating {note_length} notes and rests over {bars} bars at {tempo} BPM in {time_signature}...")
            for midi_note_number in random.sample(piano_midi_notes_all, k=12):
                print(f"MIDI note number = {midi_note_number}")
                print(f"MIDI note name = {number_to_note(midi_note_number)}")
                midi_sequence = []
                for note_index in range(total_notes_per_sequence):
                    if note_index % 2 == 0:
                        midi_sequence.append(Message("note_on", note=midi_note_number, time=(note_length_in_seconds * note_index)))
                        # 1/8 > ((3/8) / 2) = 0.125 > 0.1875 = False
                        # 0   1   2   3   4   5
                        # on  off on  off on  off
                    else:
                        midi_sequence.append(Message("note_off", note=midi_note_number, time=(note_length_in_seconds * note_index)))
                midi_sequence_duration = len(midi_sequence) * note_length_in_seconds
                print(f"MIDI sequence duration = {midi_sequence_duration} seconds")
                print(f"Generating audio for {len([i for i in range(int(float(bars) * notes_per_bar))])} alternating {note_length} notes and rests over {bars} bars at {tempo} BPM in {time_signature}...")
                audio = talunolxv2(midi_messages=midi_sequence, duration=midi_sequence_duration, sample_rate=sample_rate, num_channels=num_channels)
                preset_name = f"talunolxv2_{talunolxv2_preset_name}"
                folder_name = f"{preset_name}_{number_to_note(midi_note_number)}_{bars}bars"
                if not os.path.exists(os.path.join('P:', 'renders', '2bars', preset_name)):
                    os.makedirs(os.path.join('P:', 'renders', '2bars', preset_name), exist_ok=True)
                render_filename = f"{folder_name}_{tempo}BPM_alternating{int(note_length.split('/')[1])}notes_{numerator}n{denominator}d.wav"
                # Save JSON parameter states, i.e. preset/patch
                param_value_dict = {parameter_name: getattr(talunolxv2, parameter_name) for parameter_name in talunolxv2.parameters.keys()}
                # Convert pedalboard._pedalboard.WrappedBool to True/False
                param_value_dict = {k: (bool(v) if isinstance(v, WrappedBool) else v) for k, v in param_value_dict.items()}
                print(f"Saving JSON parameter dict: {render_filename[0:-4]}_params.json")
                with open(os.path.join('P:', 'renders', '2bars', preset_name, f"{render_filename[0:-4]}_params.json"), 'w') as file:
                    json.dump(param_value_dict, file, indent=4, ensure_ascii=True)
                # save raw state
                print(f"Saving raw plugin state binary file: {render_filename[0:-4]}_params.bin")
                with open(os.path.join('P:', 'renders', '2bars', preset_name, f"{render_filename[0:-4]}_params.bin"), 'wb') as file:
                    file.write(talunolxv2.raw_state)
                print(f"Saving audio render @ {sample_rate} Hz and depth of 32-bit floating point across {num_channels} channels...")
                file_path = os.path.join('P:', 'renders', '2bars', preset_name, render_filename)
                with pb.io.AudioFile(file_path, "w", sample_rate, num_channels=num_channels, bit_depth=32) as f:
                    f.write(audio)
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
>>> for filter_1_type in tqdm(serum2.filter_1_type.valid_values, unit='filter_1_type'):
...     serum2.filter_1_type = filter_1_type
...     wavetable_path = decrypted_preset_data["Oscillator0"]['WTOsc0']['relativePathToWT']
...     wavetable_name_stripped = re.search(r'/(.*)\.wav', wavetable_path)[1].replace(' ', '')
...     for wavetable_position in tqdm([1, 3, 4, 5], unit='wavetable_position'):
...         serum2.a_wt_pos = float(wavetable_position)
...         esc_f1type = re.sub(r'[^a-zA-Z0-9+-]', '', serum2.filter_1_type)
...         for midi_note_number in tqdm(random.sample(range(21,109), k=8), unit='midi_note_number'):
...             Path(os.path.join(
...                 os.path.curdir,
...                 "renders",
...                 "random",
...                 "Serum2",
...                 f"{wavetable_name_stripped}_{serum2.a_wt_pos}",
...                 f"{number_to_note(midi_note_number)}_{midi_note_number}MIDI",
...                 f"{esc_f1type}f1type"
...             )).mkdir(parents=True, exist_ok=True)
...             midi_sequence = []
...             midi_sequence.append(Message("note_on", note=midi_note_number, time=0))
...             midi_sequence.append(Message("note_off", note=midi_note_number, time=note_length_in_seconds))
...             k = int(0.1 * len(serum2.filter_1_freq_hz.valid_values))
...             for filter_1_freq_hz in tqdm(random.sample(serum2.filter_1_freq_hz.valid_values, k=k), unit='filter_1_freq_hz'):
...                 serum2.filter_1_freq_hz = filter_1_freq_hz
...                 print(f"Generating audio with wavetable {wavetable_path} in position {serum2.a_wt_pos}")
...                 print(f"at pitch {number_to_note(midi_note_number)} = {midi_note_number} MIDI for {note_length} note at {tempo} BPM in {time_signature}")
...                 print(f"with filter1type {serum2.filter_1_type} @ {serum2.filter_1_freq_hz} Hz...")
...                 audio = serum2(midi_messages=midi_sequence, duration=(note_length_in_seconds*1.1), sample_rate=sample_rate, num_channels=num_channels)
...                 param_value_dict = {parameter_name: getattr(serum2, parameter_name) for parameter_name in serum2.parameters.keys()}
...                 param_value_dict = {k: (bool(v) if isinstance(v, WrappedBool) else v) for k, v in param_value_dict.items()}
...                 preset_name = f"Serum2_{wavetable_name_stripped}_{serum2.a_wt_pos}_{number_to_note(midi_note_number)}_{midi_note_number}MIDI_{esc_f1type}f1type_{serum2.filter_1_freq_hz}f1hz"
...                 render_filename = f"{preset_name}.wav"
...                 json_file_path = os.path.join(
...                     os.path.curdir,
...                     "renders",
...                     "random",
...                     "Serum2",
...                     f"{wavetable_name_stripped}_{serum2.a_wt_pos}",
...                     f"{number_to_note(midi_note_number)}_{midi_note_number}MIDI",
...                     f"{esc_f1type}f1type",
...                     f"{render_filename[0:-4]}_params.json"
...                 )
...                 with open(json_file_path, 'w') as file:
...                     json.dump(param_value_dict, file, indent=4, ensure_ascii=True)
...                 raw_state_bin_file_path = os.path.join(
...                     os.path.curdir,
...                     "renders",
...                     "random",
...                     "Serum2",
...                     f"{wavetable_name_stripped}_{serum2.a_wt_pos}",
...                     f"{number_to_note(midi_note_number)}_{midi_note_number}MIDI",
...                     f"{esc_f1type}f1type",
...                     f"{render_filename[0:-4]}_params.bin"
...                 )
...                 with open(raw_state_bin_file_path, 'wb') as file:
...                     file.write(serum2.raw_state)
...                 wav_file_path = os.path.join(
...                     os.path.curdir,
...                     "renders",
...                     "random",
...                     "Serum2",
...                     f"{wavetable_name_stripped}_{serum2.a_wt_pos}",
...                     f"{number_to_note(midi_note_number)}_{midi_note_number}MIDI",
...                     f"{esc_f1type}f1type",
...                     render_filename
...                 )
...                 with pb.io.AudioFile(wav_file_path, "w", sample_rate, num_channels=num_channels, bit_depth=32) as f:
...                     f.write(audio)
