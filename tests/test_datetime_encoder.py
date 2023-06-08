import numpy as np
import pandas as pd
from alpha_automl.builtin_primitives import (CyclicalFeature,
                                             Datetime64ExpandEncoder,
                                             DummyEncoder)


class TestDatetimeEncoder:
    """This is the testcases for datetime encoder."""

    df = pd.DataFrame(
        data={
            "col1": ["12/1/2009 23:19", "2022-06-30 18:37:00", "2025-09-24 20:37:00"]
        },
        index=[0, 1, 2],
    )

    def test_cyclical_feature(self):
        encoder = CyclicalFeature()
        encoder.fit(self.df)
        np_array = encoder.transform(self.df)
        assert np_array.shape == (len(self.df), 16)
    def test_datetime64_expand_encoder(self):
        encoder = Datetime64ExpandEncoder()
        encoder.fit(self.df)
        np_array = encoder.transform(self.df)
        assert (np_array[0] == np.array([2009, 12, 1, 23, 19, 1, 335, 4])).all()
    def test_dummy_encoder(self):
        encoder = DummyEncoder()
        encoder.fit(self.df)
        np_array = encoder.transform(self.df)
        print(np_array)
        assert np_array.shape == (len(self.df), 4 * len(self.df) + 4)
