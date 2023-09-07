import numpy as np
import torch
import logging

from sklearn.feature_extraction import image
import skimage
from skimage.feature import ORB, fisher_vector, hog, learn_gmm

from alpha_automl.base_primitive import BasePrimitive
from alpha_automl._optional_dependency import import_optional_dependency

import torch.nn.functional as F

clip = import_optional_dependency('clip')

logger = logging.getLogger("IMAGE_ENCODER")


class RGB2GrayTransformer(BasePrimitive):
    """
    Convert an array of RGB images to grayscale
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])


class CLIPTransformer(BasePrimitive):
    """
    Convert an array of RGB images to grayscale
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _ = clip.load("ViT-B/32", device=self.device)
        pass

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        def clip(img):
            # img = np.transpose(img,(2,0,1))
            img = torch.from_numpy(img)
            img = img[None, :, :, :]
            img = F.interpolate(img, (224, 224))
            img = self.model.encode_image(img)
            img = torch.squeeze(img)
            return img.detach().cpu().numpy()
        return np.array([clip(img) for img in X])


class HogTransformer(BasePrimitive):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """

    def __init__(
        self,
        y=None,
        orientations=9,
        pixels_per_cell=(14, 14),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    ):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.rgb2grey = RGB2GrayTransformer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        def local_hog(X):
            return hog(
                X,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
            )

        X = self.rgb2grey.transform(X)
        return np.array([local_hog(img) for img in X])


class FisherVectorTransformer(BasePrimitive):
    """
    Convert an array of RGB images to grayscale
    """

    def __init__(self, n_keypoints=5, harris_k=0.01, k=16):
        self.n_keypoints = n_keypoints
        self.harris_k = harris_k
        self.k = k
        self.rgb2grey = RGB2GrayTransformer()

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        X = self.rgb2grey.transform(X)
        descriptors = []
        for x in X:
            detector_extractor = ORB(
                n_keypoints=self.n_keypoints, harris_k=self.harris_k
            )
            detector_extractor.detect_and_extract(x)
            descriptors.append(detector_extractor.descriptors.astype("float32"))

        gmm = learn_gmm(descriptors, n_modes=self.k)

        fvs = np.array(
            [fisher_vector(descriptor_mat, gmm) for descriptor_mat in descriptors]
        )
        return fvs


class SkPatchExtractor(BasePrimitive):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        self.extractor = image.PatchExtractor()
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return self.extractor.transform(X).reshape((X.shape[0], -1))