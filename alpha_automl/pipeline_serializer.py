from sklearn.pipeline import Pipeline as Pipeline


class PipelineSerializer(Pipeline):

    def __init__(self, pipeline_sk, label_encoder=None):
        self.label_encoder = label_encoder
        super().__init__(pipeline_sk.steps)

    def fit(self, X, y):
        if self.label_encoder is not None:
            y = self.label_encoder.fit_transform(y)
        super().fit(X, y)

    def predict(self, X):
        predictions = super().predict(X)

        if self.label_encoder is not None:
            return self.label_encoder.inverse_transform(predictions)
        else:
            return predictions

    def predict_proba(self, X):
        return super().predict_proba(X)
