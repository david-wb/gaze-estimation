from util.eye_sample import EyeSample


class EyePrediction():
    def __init__(self, eye_sample: EyeSample, landmarks):
        self._eye_sample = eye_sample
        self._landmarks = landmarks

    @property
    def eye_sample(self):
        return self._eye_sample

    @property
    def landmarks(self):
        return self._landmarks
