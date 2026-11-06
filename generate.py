import pickle, numpy as np, random
from music21 import stream, note, chord, instrument, midi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation

# Load data
with open('notes.pkl', 'rb') as f:
    notes = pickle.load(f)
with open('pitchnames.pkl', 'rb') as f:
    pitchnames = pickle.load(f)

n_vocab = len(pitchnames)
note_to_int = dict((n, i) for i, n in enumerate(pitchnames))
int_to_note = dict((i, n) for i, n in enumerate(pitchnames))

sequence_length = 50

# Rebuild same model
model = Sequential()
model.add(LSTM(256, input_shape=(sequence_length, 1), return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(256))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.load_weights('weights_best.hdf5')

# Generate 200 notes
start = random.randint(0, len(notes) - sequence_length)
pattern = [note_to_int[n] for n in notes[start:start+sequence_length]]
prediction_output = []

for i in range(200):
    x = np.reshape(pattern, (1, len(pattern), 1)) / float(n_vocab)
    prediction = model.predict(x, verbose=0)
    index = np.argmax(prediction)
    result = int_to_note[index]
    prediction_output.append(result)
    pattern.append(index)
    pattern = pattern[1:]

# Convert sequences to MIDI
offset = 0
output_notes = []
for pattern in prediction_output:
    if pattern == 'REST':
        offset += 0.5
    elif ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        chord_notes = []
        for n in notes_in_chord:
            new_note = note.Note(int(n))
            new_note.storedInstrument = instrument.Piano()
            chord_notes.append(new_note)
        new_chord = chord.Chord(chord_notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
        offset += 0.5
    else:
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)
        offset += 0.5

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp='generated_music.mid')
print("Saved: generated_music.mid - Play it in any player!")

# To convert to .wav / .mp3 audio, use: https://onlinesequencer.net or fluidsynth
# Command: fluidsynth -ni soundfont.sf2 generated_music.mid -F generated_music.wav
