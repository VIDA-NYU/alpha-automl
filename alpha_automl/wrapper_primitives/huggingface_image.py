import numpy as np
import torch
from alpha_automl.base_primitive import BasePrimitive
from alpha_automl.builtin_primitives.image_encoder import ImageReader
from alpha_automl._optional_dependency import check_optional_dependency

ml_task = 'image'
check_optional_dependency('transformers', ml_task)
from transformers import AutoModel, AutoFeatureExtractor

DEFAULT_MODEL_ID = 'openai/clip-vit-base-patch32'


class HuggingfaceImageTransformer(BasePrimitive):
    """
    Convert an array of RGB images to grayscale
    """

    def __init__(self, model_id=DEFAULT_MODEL_ID):
        self.model_id = model_id
        self.reader = ImageReader(width=224, height=224)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.processor = AutoFeatureExtractor.from_pretrained(self.model_id)

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        X = self.reader.transform(X)
        batch_size = 32
        total_length = len(X)
        batch_embeddings = []

        X = np.transpose(X, (0, 3, 1, 2))

        def non_clip(img):
            ids = self.processor(batch_img, return_tensors='pt')
            self.model.to(device)
            ids = ids.to(device)
            self.model.to(device)
            ids = ids.to(device)
            self.model.eval()
            with torch.no_grad():
                out = self.model(**ids)

            return np.squeeze(out.pooler_output.cpu().numpy())

        def clip(img):
            img = torch.from_numpy(img)
            img = img.float()
            img = self.model.get_image_features(img)
            return img.detach().numpy()

        steps = total_length // batch_size

        for start in range(0, total_length, batch_size):
            if start == (steps * batch_size):
                batch_img = X[start:total_length]

            else:
                batch_img = X[start : start + batch_size]

            if hasattr(self.model, 'get_image_features'):
                image_embedding = clip(batch_img)
            else:
                image_embedding = non_clip(batch_img)
            torch.cuda.empty_cache()

            if start == 0:
                batch_embeddings = image_embedding
            else:
                batch_embeddings = np.concatenate(
                    (batch_embeddings, image_embedding), axis=0
                )

        return batch_embeddings
