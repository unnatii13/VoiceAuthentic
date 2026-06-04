import os
import matplotlib.pyplot as plt
from scipy.io import wavfile

REAL_AUDIO = "dataset/real/LJ001-0001.wav"
FAKE_AUDIO = "dataset/fake/LJ001-0004_gen.wav"


def generate_waveform(audio_path, output_path, title):
    sample_rate, data = wavfile.read(audio_path)

    plt.figure(figsize=(12, 4))

    plt.plot(data)

    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")

    plt.tight_layout()

    plt.savefig(output_path)

    plt.show()


generate_waveform(
    REAL_AUDIO,
    "outputs/waveforms/real_waveform.png",
    "Real Audio Waveform"
)

generate_waveform(
    FAKE_AUDIO,
    "outputs/waveforms/fake_waveform.png",
    "Fake Audio Waveform"
)

print("Waveforms Generated Successfully")