import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


REAL_AUDIO = "dataset/real/LJ001-0001.wav"
FAKE_AUDIO = "dataset/fake/LJ001-0004_gen.wav"


def generate_spectrogram(audio_path, output_path, title):

    sample_rate, data = wavfile.read(audio_path)

    plt.figure(figsize=(10, 5))

    plt.specgram(data, Fs=sample_rate)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()

    print(f"{title} Saved Successfully")


generate_spectrogram(
    REAL_AUDIO,
    "outputs/spectrograms/real_spectrogram.png",
    "Real Audio Spectrogram"
)

generate_spectrogram(
    FAKE_AUDIO,
    "outputs/spectrograms/fake_spectrogram.png",
    "Fake Audio Spectrogram"
)

print("\nSpectrogram Generation Completed")