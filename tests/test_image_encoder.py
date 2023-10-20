import os

import numpy as np
import pandas as pd

from alpha_automl.builtin_primitives.image_encoder import (
    CannyEdgeDetection,
    FisherVectorTransformer,
    HogTransformer,
    ImageReader,
    RGB2GrayTransformer,
    SkPatchExtractor,
    ThresholdOtsu,
)

from alpha_automl.wrapper_primitives.clip import HuggingfaceImageTransformer


class TestImageEncoder:
    dataset = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            "test_data/digits/digits.csv",
        )
    )
    dataset["image"] = dataset["image"].apply(
        lambda x: os.path.join(
            os.path.join(os.path.dirname(__file__), "test_data/digits/media"),
            x,
        )
    )
    X = dataset[["image"]]
    y = dataset[["label"]]

    def test_image_reader(self):
        reader = ImageReader()
        im = reader.transform(self.X)
        assert im.shape == (10, 80, 80, 3)

    def test_rgb_2_grey_transformer(self):
        rgb2grey = RGB2GrayTransformer()
        im = rgb2grey.transform(self.X)

        assert im.shape == (10, 6400)

    def test_hog_transformer(self):
        hog = HogTransformer()
        im = hog.transform(self.X)

        assert im.shape == (10, 576)

    def test_fisher_vector_transformer(self):
        fisher = FisherVectorTransformer()
        im = fisher.transform(self.X)

        assert im.shape == (10, 8208)

    def test_sk_patch_extractor(self):
        patch = SkPatchExtractor()
        im = patch.transform(self.X)

        assert im.shape == (10, 1023168)

    def test_threshold_otsu(self):
        threshold = ThresholdOtsu()
        im = threshold.transform(self.X)

        assert im.shape == (10, 6400)

    def test_canny_edge_detection(self):
        canny = CannyEdgeDetection()
        im = canny.transform(self.X)

        assert im.shape == (10, 6400)

    def test_huggingface_CLIP_transformer(self):
        clip = HuggingfaceImageTransformer()
        im = clip.transform(self.X)

        assert im.shape == (10, 512)