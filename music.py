from music21 import converter, instrument, note, chord, stream
from fractions import Fraction
import glob
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
import numpy as np

files = glob.glob("midis/ragtime/*.mid")
notes = []
TIMESTEP = 0.25

def clean_chord(chrd):
    chnotes = chrd.split('.')
    foundIndex = -1
    for i in range(len(chnotes) - 1):
        for j in range(i + 1, len(chnotes)):
            if chnotes[i] == chnotes[j]:
                foundIndex = j
    if foundIndex > 0:
        del chnotes[foundIndex]
    return '.'.join(chnotes)

for f in files:
    midi = converter.parse(f)

    parts = instrument.partitionByInstrument(midi)

    if parts: # file has instrument parts
        notes_to_parse = parts.parts[0].recurse()
    else: # file has notes in a flat structure
        notes_to_parse = midi.flat.notes

    
    prev_offset = 0.0
    for element in notes_to_parse:
        if not isinstance(element, note.Note) and not isinstance(element, chord.Chord):
            continue
        duration = element.duration.quarterLength
        try:
            duration = float(duration)
        except:
            print('fraction:', Fraction(duration))
            duration = Fraction(duration)
        
        if isinstance(element, note.Note):
            name = str(element.pitch.name)
        elif isinstance(element, chord.Chord):
            name = '.'.join(str(n.name) for n in element.pitches)
            name = clean_chord(name)
        #notes.append(f"{name}${duration}")
        notes.append(name + "$" + str(duration))
        #notes.append(name)

        restnotes = int((element.offset - prev_offset) / TIMESTEP - 1)
        prev_offset = element.offset
        for _ in range(restnotes):
            notes.append("NULL")

pitchnames = sorted(set(item for item in notes))
sequence_length = 100
print(notes)
print(len(pitchnames))

exit()

note_to_int = dict((n, number) for number, n in enumerate(pitchnames))

int_to_note = dict((note_to_int[n], n) for n in note_to_int)
n_vocab = len(pitchnames)


model = keras.Sequential()
model.add(Bidirectional(LSTM(512, return_sequences=True),
    input_shape= (sequence_length, 1)
))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(512)))
model.add(Dense(n_vocab, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop")

x = []
y = []

for i in range(len(notes) - sequence_length):
    seq_in = notes[i:i + sequence_length]
    seq_out = notes[i + sequence_length]
    x.append([note_to_int[n] for n in seq_in])
    y.append(note_to_int[seq_out])

y = keras.utils.to_categorical(y)

x = np.reshape(x, (len(x), sequence_length, 1))
x = x / n_vocab

cp_callback = keras.callbacks.ModelCheckpoint(filepath="./midis/weights/cp.ckpt",
                                                 save_weights_only=True,
                                                 verbose=1)

#model.load_weights('./midis/weights/cp.ckpt')
#model.fit(x, y, epochs=2, callbacks=[cp_callback])

start = np.random.randint(0, len(x) - 1)

pattern = x[start]
song = []

for idx in range(500):
    prediction = model.predict(np.reshape(pattern, (1, len(pattern), 1)))
    int_note = np.argmax(prediction)
    notestr = int_to_note[int_note]
    song.append(notestr)
    pattern = np.append(pattern, int_note / n_vocab)
    pattern = pattern[1:len(pattern)]

offset = 0.0
output_notes = []

for item in song:
    print(item)
    if "NULL" in item:
        offset += TIMESTEP
        continue

    (notestr, duration) = item.split('$')
    duration = float(duration)
    if '.' in notestr:
        chord_notes = notestr.split('.')
        notes = []
        for c_note in chord_notes:
            myNote = note.Note(c_note)
            myNote.storedInstrument = instrument.Piano()
            notes.append(myNote)
        new_chord = chord.Chord(notes, quarterLength=duration)
        new_chord.offset = offset
        output_notes.append(new_chord)
    elif "NULL" not in notestr:
        myNote = note.Note(notestr, quarterLength=duration)
        myNote.storedInstrument = instrument.Piano()
        myNote.offset = offset
        output_notes.append(myNote)

    offset += TIMESTEP

print(output_notes)

midi_stream = stream.Stream(output_notes)
midi_stream.write('midi', fp = 'midis/BB_test_output.mid')