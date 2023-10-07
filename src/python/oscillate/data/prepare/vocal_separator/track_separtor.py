import os
import librosa
from spleeter.separator import Separator


class TrackSeparator:
    def __init__(self, input_folder, vocal_folder, instrumental_folder):
        self.input_folder = input_folder
        self.vocal_folder = vocal_folder
        self.instrumental_folder = instrumental_folder
        self.separator = Separator('spleeter:2stems')

    def separate_tracks(self):
        for filename in os.listdir(self.input_folder):
            if filename.endswith(".wav"):
                audio_loader = librosa.load(os.path.join(self.input_folder, filename), sr=44100, mono=True)
                prediction = self.separator.separate(audio_loader)

                librosa.output.write_wav(os.path.join(self.vocal_folder, filename), prediction['vocals'], sr=44100)
                librosa.output.write_wav(os.path.join(self.instrumental_folder, filename), prediction['accompaniment'], sr=44100)

