"""
This predict.py file is not actively used.
It exists only to pass cog validation requirements.
The actual functionality is in train.py
"""
from cog import BasePredictor, Path, Input

class Predictor(BasePredictor):
    def setup(self):
        pass

    def predict(self,
            prompt: str = Input(description="Temporary predict is unused"),
    ) -> Path:
        pass
