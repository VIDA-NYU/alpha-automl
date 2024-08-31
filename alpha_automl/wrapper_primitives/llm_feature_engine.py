import os
import ast
import copy
import logging
import numpy as np
import pandas as pd
import openai

from alpha_automl.base_primitive import BasePrimitive

logger = logging.getLogger(__name__)

class LLMFeatureGenerator(BasePrimitive):
    def __init__(self, description=None, extra_system_prompt=None):
        self.description = description
        self.extra_system_prompt = extra_system_prompt
        self.prompt = None
        self.code = None
        pass

    def fit(self, X, y=None):
        self.prompt = build_prompt_from_df(description=self.description, df=X)
        self.code = generate_code(self.prompt, self.extra_system_prompt)
        return self

    def transform(self, X, y=None):
        X_cp = copy.deepcopy(X)
        loc = {}
        access_scope = {"df": X_cp, "pd": pd, "np": np}
        parsed = ast.parse(self.code)
        exec(compile(parsed, filename="<ast>", mode="exec"), access_scope, loc)
        return X_cp

def get_prompt(
    df, description, iterative=1, data_description_unparsed=None, samples=None, **kwargs
):
    how_many = (
        "up to 10 useful columns. Generate as many features as useful for downstream classifier, but as few as necessary to reach good performance."
        if iterative == 1
        else "exactly one useful column"
    )
    return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{data_description_unparsed}"

Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}
    
This code was written by an expert datascientist working to improve predictions. It is a snippet of code that adds new columns to the dataset.
Number of samples (rows) in training dataset: {int(len(df))}
    
This code generates additional columns that are useful for a downstream classification algorithm (such as XGBoost) predicting \"{description}\".
Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes.
This code also drops columns, if these may be redundant and hurt the predictive performance of the downstream classifier (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small.
The classifier will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is accuracy. The best performing code will be selected.
Added columns can be used in other codeblocks, dropped columns are not available anymore.
Remember do not include np.inf, -np.inf in any generated columns.

Code formatting for each added column:
```python
# (Feature name and description)
# Usefulness: (Description why this adds useful real world knowledge to classify \"{description}\" according to dataset description and attributes.)
# Input samples: (Three samples of the columns used in the following code, e.g. '{df.columns[0]}': {list(df.iloc[:3, 0].values)}, '{df.columns[1]}': {list(df.iloc[:3, 1].values)}, ...)
(Some pandas code using {df.columns[0]}', '{df.columns[1]}', ... to add a new column for each row in df)
```end

Code formatting for dropping columns:
```python
# Explanation why the column XX is dropped
df.drop(columns=['XX'], inplace=True)
```end

Each codeblock generates {how_many} and can drop unused columns (Feature selection).
Each codeblock ends with ```end and starts with "```python"
Codeblock:
"""

def build_prompt_from_df(description, df, iterative=1):
    data_description_unparsed = description
    feature_importance = {}  # xgb_eval(_obj)

    samples = ""
    df_ = df.head(3).iloc[:, :1000]
    for i in list(df_):
        # show the list of values
        nan_freq = "%s" % float("%.2g" % (df[i].isna().mean() * 100))
        s = df_[i].tolist()
        if str(df[i].dtype) == "float64":
            s = [round(sample, 2) for sample in s]
        samples += (
            f"{df_[i].name} ({df[i].dtype}): NaN-freq [{nan_freq}%], Samples {s}\n"
        )

    kwargs = {
        "data_description_unparsed": data_description_unparsed,
        "samples": samples,
        "feature_importance": {
            k: "%s" % float("%.2g" % feature_importance[k]) for k in feature_importance
        },
    }

    prompt = get_prompt(
        df,
        description,
        data_description_unparsed=data_description_unparsed,
        iterative=iterative,
        samples=samples,
    )

    return prompt

def generate_code(prompt, extra_system_prompt, model="gpt-4o"):
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    client = openai.OpenAI(api_key=openai_api_key)
    messages = [
        {
            "role": "system",
            "content": "You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible.",
        },
    ]
    if extra_system_prompt is not None:
        messages.append(
            {
                "role": "system",
                "content": extra_system_prompt,
            }
        )
    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    if model == "skip":
        return ""

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stop=["```end"],
        temperature=0.5,
        max_tokens=4096,
    )
    code = completion.choices[0].message.content
    code = code.replace("```python", "").replace("```", "").replace("<end>", "")
    return code
