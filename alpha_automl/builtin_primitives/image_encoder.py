import logging
import numpy as np
import pandas as pd
from alpha_automl._optional_dependency import check_optional_dependency
from alpha_automl.base_primitive import BasePrimitive

ml_task = 'image'
check_optional_dependency('skimage', ml_task)

from skimage.color import gray2rgb, rgb2gray, rgba2rgb
from skimage.feature import ORB, canny, fisher_vector, hog, learn_gmm
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.transform import resize
from sklearn.feature_extraction import image

logging.getLogger("PIL").setLevel(logging.CRITICAL + 1)
logger = logging.getLogger("automl")


class ImageReader(BasePrimitive):
    """Convert Image path to numpy array"""

    def __init__(self, width=80, height=80):
        self.width = width
        self.height = height

    def fit(self, X, y=None):
        return self

    def transform(self, images):
        data = []
        if isinstance(images, pd.DataFrame):
            for file in images[images.columns[0]]:
                im = imread(file)
                im = resize(im, (self.width, self.height))
                if len(im.shape) < 3:
                    im = gray2rgb(im)
                elif im.shape[2] == 4:
                    im = rgba2rgb(im)
                elif im.shape[2] != 3:
                    im = gray2rgb(im[:, :, 0])
                data.append(im)
        else:
            for file in images:
                im = imread(file[0])
                im = resize(im, (self.width, self.height))
                if len(im.shape) < 3:
                    im = gray2rgb(im)
                elif im.shape[2] == 4:
                    im = rgba2rgb(im)
                elif im.shape[2] != 3:
                    im = gray2rgb(im[:, :, 0])
                data.append(im)
        return np.array(data)


class ThresholdOtsu(BasePrimitive):
    """
    Filter image with a calculated threshold
    """

    def __init__(self):
        self.reader = ImageReader()
        pass

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        X = self.reader.transform(X)

        def threshold(img):
            img = rgb2gray(img)
            threashold_value = threshold_otsu(img)
            img = img > threashold_value
            return img.flatten()

        return np.array([threshold(img) for img in X])


class CannyEdgeDetection(BasePrimitive):
    """
    Filter image with canny edge detection
    """

    def __init__(self):
        self.reader = ImageReader()
        pass

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        X = self.reader.transform(X)

        def canny_edge(img):
            img = rgb2gray(img)
            img = canny(img)
            return img.flatten()

        return np.array([canny_edge(img) for img in X])


class RGB2GrayTransformer(BasePrimitive):
    """
    Convert an array of RGB images to grayscale
    """

    def __init__(self):
        self.reader = ImageReader()
        pass

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        X = self.reader.transform(X)
        return np.array([rgb2gray(img).flatten() for img in X])


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
        self.reader = ImageReader()

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

        X = self.reader.transform(X)
        X = np.array([rgb2gray(img) for img in X])
        return np.array([local_hog(img) for img in X])


class FisherVectorTransformer(BasePrimitive):
    """
    Fisher vector is an image feature encoding and quantization technique
    that can be seen as a soft or probabilistic version of the popular
    bag-of-visual-words or VLAD algorithms
    """

    def __init__(self, n_keypoints=5, harris_k=0.01, k=16):
        self.n_keypoints = n_keypoints
        self.harris_k = harris_k
        self.k = k
        self.reader = ImageReader()
        self.gmm = None

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        X = self.reader.transform(X)
        X = np.array([rgb2gray(img) for img in X])
        descriptors = []
        for x in X:
            detector_extractor = ORB(
                n_keypoints=self.n_keypoints, harris_k=self.harris_k
            )
            detector_extractor.detect_and_extract(x)
            descriptors.append(detector_extractor.descriptors.astype("float32"))

        if self.gmm is None:
            self.gmm = learn_gmm(descriptors, n_modes=self.k)

        fvs = np.array(
            [fisher_vector(descriptor_mat, self.gmm) for descriptor_mat in descriptors]
        )
        return fvs


class SkPatchExtractor(BasePrimitive):
    """
    Extracts patches from a collection of images
    """

    def __init__(self):
        self.reader = ImageReader()
        self.extractor = image.PatchExtractor()
        pass

    def fit(self, X, y=None):
        """returns itself"""
        return self

    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        X = self.reader.transform(X)
        return self.extractor.transform(X).reshape((X.shape[0], -1))
