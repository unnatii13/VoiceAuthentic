import librosa
import librosa.display
import matplotlib.pyplot as plt


REAL_AUDIO = "dataset/real/LJ001-0001.wav"
FAKE_AUDIO = "dataset/fake/LJ001-0004_gen.wav"


def generate_mfcc(audio_path, output_path, title):

    audio, sample_rate = librosa.load(audio_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=13
    )

    plt.figure(figsize=(10, 5))

    librosa.display.specshow(
        mfcc,
        x_axis="time"
    )

    plt.colorbar()

    plt.title(title)

    plt.tight_layout()

    plt.savefig(output_path)

    plt.close()

    print(f"{title} Saved Successfully")


generate_mfcc(
    REAL_AUDIO,
    "outputs/mfcc/real_mfcc.png",
    "Real Audio MFCC"
)

generate_mfcc(
    FAKE_AUDIO,
    "outputs/mfcc/fake_mfcc.png",
    "Fake Audio MFCC"
)

print("\nMFCC Generation Completed")