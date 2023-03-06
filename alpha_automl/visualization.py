# PipelineProfiler is imported inside of the function because it raises errors when is running from
# non-Jupyter/Colab environments (e.g. terminal scripts).


def plot_comparison_pipelines(pipelines, primitive_types=None):
    import PipelineProfiler
    PipelineProfiler.plot_pipeline_matrix(pipelines, primitive_types)
