import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pretty_midi
import io
from scipy.io import wavfile
from tensorflow import keras
from music21 import chord, note, converter, instrument, stream
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model

generated = None

st.title("Generate Midi with Artificial Neural Networks")

ex1 = open('MIDIwithANN1.mp3', 'rb')
ex1_bytes = ex1.read()
ex2 = open('MIDIwithANN2.mp3', 'rb')
ex2_bytes = ex2.read()
ex3 = open('MIDIwithANN3.mp3', 'rb')
ex3_bytes = ex3.read()

ex1.close()
ex2.close()
ex3.close()


def about():
    st.write('''
    Auto Generate Midi With ANN
    
    This App is Built with Keras and Streamlit''')


def examples():
    st.title("Audio Examples Generated with Model:")
    st.markdown("<div class='title blue'>All Examples Have Been Exported And "
                "Run Through FL Studio To Use Unique Instruments</div>", unsafe_allow_html=True)
    st.markdown("<div class='title blue'>The files also received randomization on their respective velocity to improve "
                "dynamics.</div>", unsafe_allow_html=True)
    st.markdown("<div class = 'title blue'> Example 1 <br></div>", unsafe_allow_html=True)
    st.audio(ex1_bytes, format='audio/mp3')
    st.markdown("<div class = 'title blue'> Example 2<br> </div>", unsafe_allow_html=True)
    st.audio(ex2_bytes, format='audio/mp3')
    st.markdown("<div class = 'title blue'> Example 3<br> </div>", unsafe_allow_html=True)
    st.audio(ex3_bytes, format='audio/mp3')


def generateMidi(midi_file):
    original_scores = [midi_file]
    original_scores = [song.chordify() for song in original_scores]
    original_chords = [[] for _ in original_scores]
    original_durations = [[] for _ in original_scores]
    original_keys = []

    for i, song in enumerate(original_scores):
        original_keys.append((str(song.analyze('key'))))

        for element in song:
            if isinstance(element, note.Note):
                original_chords[i].append(element.pitch)
                original_durations[i].append(element.duration.quarterLength)
            elif isinstance(element, chord.Chord):
                original_chords[i].append('.'.join(str(n) for n in element.pitches))
                original_durations[i].append(element.duration.quarterLength)

    unique_chords = np.unique([i for s in original_chords for i in s])
    chord_to_int = dict(zip(unique_chords, list(range(0, len(unique_chords)))))

    st.write("There are {} unique chords in this file.".format(len(unique_chords)))

    if len(unique_chords) < 100:
        st.write("This shouldn't take too long.")
        if st.button("Cancel"):
            st.stop()
    elif 100 < len(unique_chords) < 200:
        st.write("This may take up to a minute.")
        if st.button("Cancel"):
            st.stop()
    elif 200 < len(unique_chords) < 300:
        st.write("This could take between two and three minutes.")
        if st.button("Cancel"):
            st.stop()
    elif len(unique_chords) > 300:
        st.write("You may be here a while.")
        if st.button("Cancel"):
            st.stop()

    unique_durations = np.unique([i for s in original_durations for i in s])
    duration_to_int = dict(zip(unique_durations, list(range(0, len(unique_durations)))))

    int_to_chord = {i: c for c, i in chord_to_int.items()}
    int_to_duration = {i: c for c, i in duration_to_int.items()}

    sequence_length = 32

    train_chords = []
    train_durations = []

    for s in range(len(original_chords)):
        chord_list = [chord_to_int[c] for c in original_chords[s]]
        duration_list = [duration_to_int[d] for d in original_durations[s]]

        for i in range(len(chord_list) - sequence_length):
            train_chords.append(chord_list[i:i + sequence_length])
            train_durations.append(duration_list[i:i + sequence_length])

    train_chords = np.array(train_chords)
    try:
        train_chords = to_categorical(train_chords).transpose(0, 2, 1)
    except ValueError:
        st.write("Please Try a Longer Midi File - Not Enough Data")
        st.stop()

    n_samples = train_chords.shape[0]
    n_chords = train_chords.shape[1]
    input_dimensions = n_chords * n_samples
    latent_dimensions = 2

    train_chords_flat = train_chords.flatten()

    encoder_input = Input(shape=input_dimensions)
    latent = Input(shape=latent_dimensions)
    encoded = Dense(latent_dimensions, activation='tanh')(encoder_input)
    decoded = Dense(input_dimensions, activation='sigmoid')(latent)
    encoder = Model(encoder_input, encoded)
    decoder = Model(latent, decoded)
    autoencoder = Model(encoder_input, decoder(encoded))
    autoencoder.compile(loss='binary_crossentropy', optimizer='rmsprop')
    train_chords_reshape = train_chords.reshape(-1, input_dimensions)
    autoencoder.fit(train_chords_reshape, train_chords_reshape, epochs=500, verbose=0)
    generated_chords = decoder(np.random.normal(size=(1, latent_dimensions))).numpy().reshape(n_chords,
                                                                                              n_samples).argmax(0)
    chord_sequence = [int_to_chord[c] for c in generated_chords]

    generatedStream = stream.Stream()

    # Append notes and chords to stream object
    for j in range(len(chord_sequence)):
        try:
            generatedStream.append(note.Note(chord_sequence[j].replace('.', ' ')))
        except:
            generatedStream.append(chord.Chord(chord_sequence[j].replace('.', ' ')))

    st.write("Generation Complete:\n\n\n")

    return generatedStream


def generate():
    global generated
    st.write("Here, you can upload a midi file and generate a similar, completely original sound using artificial "
             "neural networks.")
    upload = st.file_uploader("Upload Midi", type=['.mid'])

    if upload is None:
        st.info("Please upload a Midi File")
        st.stop()

    # midi_data = pretty_midi.PrettyMIDI(upload)
    # audio_data = midi_data.fluidsynth()
    #
    # audio_data = np.int16(audio_data / np.max(np.abs(
    #     audio_data)) * 32767 * 0.9)
    #
    # virtualfile = io.BytesIO()
    # wavfile.write(virtualfile, 44100, audio_data)
    #
    # st.audio(virtualfile)

    midi_file = converter.parseData(upload.read())

    st.write("Your Original Audio:")
    st.write("\n\n\n")
    # st.audio(virtual_file)
    st.write("\n\n\n")
    st.write("Audio Player Here")
    st.write("\n\n\n")

    if st.button("Generate!"):
        st.write("Generating now...\n\n")
        generated = generateMidi(midi_file)
        generated.show('midi')
    if generated is not None:
        return generated


def main():
    st.image('midi.jpg')

    t = "<h2 class='title blue'> Generate Midi with Artificial Neural Networks </h2>"

    st.markdown(t, unsafe_allow_html=True)

    te = "<div class='title blue'>Built with Keras and Streamlit.io</div>"
    st.write("\n")

    st.markdown(te, unsafe_allow_html=True)

    activities = ["Home", "About", "Examples", "Generate"]

    choice = st.sidebar.selectbox("Navigation", activities)

    if choice == "About":
        about()
    if choice == "Examples":
        examples()
    if choice == "Generate":
        downloadable = generate()


if __name__ == "__main__":
    main()
