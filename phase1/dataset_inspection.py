import os
from scipy.io import wavfile

REAL_PATH = "dataset/real"
FAKE_PATH = "dataset/fake"


def inspect_folder(folder_path, label):
    print(f"\n{'=' * 50}")
    print(f"{label} DATASET")
    print(f"{'=' * 50}")

    files = os.listdir(folder_path)

    print(f"Total Files: {len(files)}\n")

    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)

            sample_rate, data = wavfile.read(file_path)

            duration = len(data) / sample_rate

            print(f"File Name    : {file}")
            print(f"Sample Rate : {sample_rate} Hz")
            print(f"Duration    : {duration:.2f} seconds")
            print(f"Samples     : {len(data)}")
            print("-" * 50)


inspect_folder(REAL_PATH, "REAL")
inspect_folder(FAKE_PATH, "FAKE")