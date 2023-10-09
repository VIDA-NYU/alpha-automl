import numpy as np
import torch

from alpha_automl._optional_dependency import import_optional_dependency
from alpha_automl.base_primitive import BasePrimitive
from alpha_automl.builtin_primitives.image_encoder import ImageReader

transformers = import_optional_dependency("transformers")

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"


class HuggingfaceCLIPTransformer(BasePrimitive):
    """
    Convert an array of RGB images to grayscale
    """

    def __init__(self, model_id=DEFAULT_MODEL_ID):
        self.model_id = model_id
        self.reader = ImageReader(width=224, height=224)
        self.model = transformers.CLIPModel.from_pretrained(self.model_id)

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        X = self.reader.transform(X)

        def clip(img):
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img)
            img = img[None, :, :, :]
            img = img.float()
            img = self.model.get_image_features(img)
            return img.detach().numpy()[0]

        return np.array([clip(img) for img in X])
