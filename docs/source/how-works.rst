How Alpha-AutoML works
=======================

Alpha-AutoML is an extensible open-source AutoML system. It leverages the reinforcement learning and neural network
components of `AlphaD3M <https://gitlab.com/ViDA-NYU/d3m/alphad3m>`__, but it relies on standard, open-source
infrastructure to specify and run pipelines. It is compatible with state-of-the-art ML techniques: by using the
Sklearn pipeline infrastructure, Alpha-AutoML is fully compatible with other standard libraries like XGBoost, Hugging
Face, Keras, PyTorch. In addition, primitives can be added on the fly through the standard Sklearn’s fit/predict API,
making it possible for Alpha-AutoML to leverage new developments in ML and keep up with the fast pace in the area.

The problem of pipeline synthesis for model discovery is framed as a single-player game
where the player iteratively builds a pipeline by selecting actions (insertion, deletion and replacement of pipeline
components). We solve the meta-learning problem using a deep neural network and a Monte Carlo tree search (MCTS).
The neural network receives as input an entire pipeline, data meta-features, and the problem, and outputs
action probabilities and estimates for the pipeline performance. The MCTS uses the network probabilities to run
simulations which terminate at actual pipeline evaluations.
To reduce the search space, we define a pipeline grammar where the rules of the grammar constitute the actions.  The
grammar rules grow linearly with the number of primitives and hence address the issue of scalability.


Support for Many ML Problems
-----------------------------

Alpha-AutoML uses a comprehensive collection of primitives provided in open-source libraries, such as scikit-learn,
XGBoost, Hugging Face, etc to derive pipelines for a wide range of machine learning tasks. These
pipelines can be applied to different data types and derive standard performance metrics.

- *Learning Tasks*: classification, regression, clustering, time series forecasting, and semi-supervised classification.
- *Data Types*: tabular, text, and image.
- *Data Formats*: CSV, Pandas dataframes, and raw images files datasets.
- *Metrics*: Standard metrics (e.g. accuracy, F1, mean squared error, etc) and custom metrics.



Usability, Model Exploration and Explanation
---------------------------------------------

Alpha-AutoML greatly simplifies the process to create predictive models. Users can interact with the system from a
Jupyter notebook, and derive models using a few lines of Python code.

Users can leverage Python-based libraries and tools to clean, transform and visualize data, as well as standard methods
to explain machine learning models.  They can also be combined to  build customized solutions for specific problems that
can be deployed to end users.

The Alpha-AutoML environment includes tools that we developed to enable users to explore the pipelines and their predictions.
PipelineProfiler is an interactive visual analytics tool that empowers data scientists to explore the pipelines derived
by Alpha-AutoML within a Jupyter notebook, and gain insights to improve them as well as make an informed decision while
selecting models for a given application.
