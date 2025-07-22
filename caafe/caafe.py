import copy
import numpy as np
from typing import Optional
from openai import OpenAI
from sklearn.model_selection import RepeatedKFold
from .caafe_evaluate import (
    evaluate_dataset,
)
from .run_llm_code import run_llm_code
from .metrics import higher_is_better


def get_prompt(
    df, ds, original_features, iterative=1, data_description_unparsed=None, more_features=True, samples=None, task="classification", metric="accuracy", **kwargs
):
    how_many = (
        "up to 5 useful columns. Generate as many features as useful for downstream downstream algorithm, but as few as necessary to reach good performance."
        if more_features == True
        else "exactly one useful column"
    )
    return f"""
The dataframe `df` is loaded and in memory. Columns are also named attributes.
Description of the dataset in `df` (column dtypes might be inaccurate):
"{data_description_unparsed}"

Columns in `df` (true feature dtypes listed here, categoricals encoded as int):
{samples}
    
This code was written by an expert datascientist working to improve predictions on the {task} task. It is a snippet of code that adds new columns to the dataset.
Number of samples (rows) in training dataset: {int(len(df))}
    
This code generates additional columns that are useful for a downstream algorithm (such as XGBoost) predicting \"{ds[4][-1]}\".
Additional columns add new semantic information, that is they use real world knowledge on the dataset. They can e.g. be feature combinations, transformations, aggregations where the new column is a function of the existing columns.
The scale of columns and offset does not matter. Make sure all used columns exist. Follow the above description of columns closely and consider the datatypes and meanings of classes.
This code also drops columns, if these may be redundant and hurt the predictive performance of the downstream algorithm (Feature selection). Dropping columns may help as the chance of overfitting is lower, especially if the dataset is small.
The downstream algorithm will be trained on the dataset with the generated columns and evaluated on a holdout set. The evaluation metric is {metric}. The best performing code will be selected.
Added columns can be used in other codeblocks, dropped columns are not available anymore.

Code formatting for each added column:
```python
# (Feature name and description)
# Feature Order: (1, 2, 3, ...), The number of original (raw) features involved in the construction of a derived feature. {original_features}
# Transformation Order: (1, 2, 3, ...), The number of sequential transformation steps required to compute the feature.
# Usefulness: (Description why this adds useful real world knowledge to classify \"{ds[4][-1]}\" according to dataset description and attributes.)
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


# Each codeblock either generates {how_many} or drops bad columns (Feature selection).


def build_prompt_from_df(ds, df, iterative=1, task="classification", metric="accuracy",more_features=True):
    data_description_unparsed = ds[-1]
    feature_importance = {}  # xgb_eval(_obj)

    samples = ""
    df_ = df.head(10)
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
        ds,
        original_features=ds[4][:-1],  # all but last element
        data_description_unparsed=data_description_unparsed,
        more_features=more_features,
        iterative=iterative,
        samples=samples,
        task=task,
        metric=metric,
    )
    return prompt


def generate_features(
    ds,
    df,
    model: str = "gpt-3.5-turbo",
    just_print_prompt: bool = False,
    iterative: int = 1,
    metric_used: Optional[str] = None,
    iterative_method: str = "logistic",
    display_method: str = "markdown",
    n_splits: int = 10,
    n_repeats: int = 2,
    task: str = "classification",
    more_features: bool = True,
):
    def format_for_display(code):
        code = code.replace("```python", "").replace("```", "").replace("<end>", "")
        return code

    if display_method == "markdown":
        from IPython.display import display, Markdown

        display_method = lambda x: display(Markdown(x))
    else:

        display_method = print

    assert (
        iterative == 1 or metric_used is not None
    ), "metric_used must be set if iterative"

    prompt = build_prompt_from_df(ds, df, iterative=iterative, task=task, metric=metric_used, more_features=more_features)

    if just_print_prompt:
        code, prompt = None, prompt
        return code, prompt, None

    def generate_code(messages):
        if model == "skip":
            return ""
        client = OpenAI()
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.5,
            max_completion_tokens=8000
        )
        
        code = completion.choices[0].message.content
        code = code.replace("```python", "").replace("```", "").replace("<end>", "").replace("end","")
        return code

    def execute_and_evaluate_code_block(full_code, code):
        old_scores, scores = [], []

        ss = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=0)
        fold = 0
        for (train_idx, valid_idx) in ss.split(df):
            fold +=1
            df_train, df_valid = df.iloc[train_idx], df.iloc[valid_idx]

            # Remove target column from df_train
            target_train = df_train[ds[4][-1]]
            target_valid = df_valid[ds[4][-1]]
            df_train = df_train.drop(columns=[ds[4][-1]])
            df_valid = df_valid.drop(columns=[ds[4][-1]])

            df_train_extended = copy.deepcopy(df_train)
            df_valid_extended = copy.deepcopy(df_valid)

            try:
                df_train = run_llm_code(
                    full_code,
                    df_train,
                    convert_categorical_to_integer=not ds[0].startswith("kaggle"),
                )
                df_valid = run_llm_code(
                    full_code,
                    df_valid,
                    convert_categorical_to_integer=not ds[0].startswith("kaggle"),
                )
                df_train_extended = run_llm_code(
                    full_code + "\n" + code,
                    df_train_extended,
                    convert_categorical_to_integer=not ds[0].startswith("kaggle"),
                )
                df_valid_extended = run_llm_code(
                    full_code + "\n" + code,
                    df_valid_extended,
                    convert_categorical_to_integer=not ds[0].startswith("kaggle"),
                )

            except Exception as e:
                display_method(f"Error in code execution. {type(e)} {e}")
                display_method(f"```python\n{format_for_display(code)}\n```\n")
                return e, None, None

            # Add target column back to df_train
            df_train[ds[4][-1]] = target_train
            df_valid[ds[4][-1]] = target_valid
            df_train_extended[ds[4][-1]] = target_train
            df_valid_extended[ds[4][-1]] = target_valid

            from contextlib import contextmanager
            import sys, os

            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    result_old = evaluate_dataset(
                        df_train=df_train,
                        df_test=df_valid,
                        prompt_id="XX",
                        name=ds[0],
                        method=iterative_method,
                        metric_used=metric_used,
                        seed=0,
                        target_name=ds[4][-1],
                    )

                    result_extended = evaluate_dataset(
                        df_train=df_train_extended,
                        df_test=df_valid_extended,
                        prompt_id="XX",
                        name=ds[0],
                        method=iterative_method,
                        metric_used=metric_used,
                        seed=0,
                        target_name=ds[4][-1],
                    )
                finally:
                    sys.stdout = old_stdout

            display_method(f"> Fold-{fold}:- Old: {result_old['score']}, New: {result_extended['score']}")
            old_scores += [result_old["score"]]
            scores += [result_extended["score"]]
        return None, scores, old_scores

    messages = [
        {
            "role": "system",
            "content": "You are an expert datascientist assistant solving Kaggle problems. You answer only by generating code. Answer as concisely as possible.",
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]
    display_method(f"# *Dataset description:*\n {ds[-1]}")

    n_iter = iterative
    full_code = ""

    i = 0
    while i < n_iter:
        try:
            code = generate_code(messages)
        except Exception as e:
            display_method("Error in LLM API." + str(e))
            continue
        i = i + 1
        e, scores, old_scores = execute_and_evaluate_code_block(
            full_code, code
        )
        if e is not None:
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Code execution failed with error: {type(e)} {e}.\n Code: ```python{code}```\n Generate next feature (fixing error?):
                                ```python
                                """,
                },
            ]
            continue

        # importances = get_leave_one_out_importance(
        #    df_train_extended,
        #    df_valid_extended,
        #    ds,
        #    iterative_method,
        #    metric_used,
        # )
        # """ROC Improvement by using each feature: {importances}"""

        old_mean = np.nanmean(old_scores)
        new_mean = np.nanmean(scores)
        sign = 1 if higher_is_better.get(metric_used, True) else -1
        improvement = sign * (new_mean - old_mean)

        add_feature = improvement > 0
        if add_feature:
            sentence = "The code was executed and changes to ´df´ were kept."
        else:
            sentence = (
                f"The last code changes to ´df´ were discarded. "
                f"(Improvement: {improvement})"
            )

        display_method(
            "\n"
            + f"*Iteration {i}*\n"
            + f"```python\n{format_for_display(code)}\n```"
        )
        display_method(f"""
- Performance before adding features {metric_used} : {old_mean}
- Performance after adding features {metric_used}  : {new_mean}
- Improvement in {metric_used}: {improvement}
- {sentence}"""
        )

        if len(code) > 10:
            messages += [
                {"role": "assistant", "content": code},
                {
                    "role": "user",
                    "content": f"""Performance after adding feature {metric_used} {new_mean}. {sentence}\n
Next codeblock:
""",
                },
            ]
        if add_feature:
            full_code += code

    return full_code, prompt, messages
