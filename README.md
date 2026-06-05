\# VoiceAuthentic: Real-Time Deepfake Voice Scammer Firewall



\## Project Overview



VoiceAuthentic is an AI-powered cybersecurity project designed to detect synthetic or AI-generated voices in real time. The objective is to protect individuals and organizations from voice cloning scams by analyzing audio signals and identifying characteristics of deepfake speech.



The project aims to serve as a defensive layer against social engineering attacks where attackers impersonate family members, executives, or trusted individuals using AI-generated voice technology.



\---



\## Phase 1: Audio Dataset Analysis and Feature Extraction



\### Objective



The goal of Phase 1 is to understand the audio dataset and extract meaningful features that can later be used for machine learning and deepfake voice detection.



\### Dataset Structure



The dataset contains two categories:



\* Real Audio Samples

\* Fake (AI-Generated) Audio Samples



Project Structure:



VoiceAuthentic/



\* dataset/



&#x20; \* real/

&#x20; \* fake/

\* phase1/



&#x20; \* dataset\_inspection.py

&#x20; \* waveform.py

&#x20; \* spectrogram.py

&#x20; \* mfcc.py

\* outputs/



&#x20; \* waveforms/

&#x20; \* spectrograms/

&#x20; \* mfcc/



\---



\## Module 1: Dataset Inspection



File: dataset\_inspection.py



\### Purpose



This module analyzes the dataset and provides:



\* Number of audio files

\* Sample rate

\* Audio duration

\* Basic dataset statistics



\### Outcome



Successfully inspected all real and fake audio samples and verified dataset integrity.



\---



\## Module 2: Waveform Analysis



File: waveform.py



\### Purpose



Waveforms visualize audio signals in the time domain.



\### Process



Audio File

→ Read Audio Samples

→ Plot Waveform

→ Save Visualization



\### Output



\* real\_waveform.png

\* fake\_waveform.png



\### Observation



Waveforms provide a visual representation of voice amplitude changes over time and help identify signal patterns.



\---



\## Module 3: Spectrogram Analysis



File: spectrogram.py



\### Purpose



Spectrograms visualize frequency content over time.



\### Process



Audio File

→ Frequency Analysis

→ Spectrogram Generation

→ Save Visualization



\### Output



\* real\_spectrogram.png

\* fake\_spectrogram.png



\### Observation



Spectrograms reveal hidden frequency patterns and are commonly used as input for deep learning models in speech processing and deepfake detection.



\---



\## Module 4: MFCC Feature Extraction



File: mfcc.py



\### Purpose



MFCC (Mel-Frequency Cepstral Coefficients) are one of the most widely used audio features in speech recognition and speaker verification systems.



\### Process



Audio File

→ MFCC Extraction

→ Feature Visualization

→ Save Visualization



\### Output



\* real\_mfcc.png

\* fake\_mfcc.png



\### Observation



MFCC features capture important characteristics of human speech and provide a compact representation suitable for machine learning models.



\---



\## Technologies Used



\* Python

\* NumPy

\* SciPy

\* Matplotlib

\* Librosa

\* Git

\* GitHub

\* IntelliJ IDEA



\---



\## Phase 1 Deliverables



Completed:



\* Dataset Inspection

\* Waveform Visualization

\* Spectrogram Visualization

\* MFCC Feature Extraction

\* GitHub Repository Setup

\* Project Documentation



\---





