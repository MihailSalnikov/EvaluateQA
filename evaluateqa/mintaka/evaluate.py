# Based on original evaluation scritp: https://github.com/amazon-science/mintaka/blob/main/evaluate/evaluate.py
import urllib.request
import ujson
import pandas as pd
import tempfile
import regex
import collections
import unicodedata
from joblib import Memory
from typing import Dict, Union, List, Optional


MEMORY = Memory(tempfile.gettempdir() + "/QALeaderboard_mintaka_data")


MINTAKA_URI = {
    "dev": "https://raw.githubusercontent.com/amazon-science/mintaka/main/data/mintaka_dev.json",
    "test": "https://raw.githubusercontent.com/amazon-science/mintaka/main/data/mintaka_test.json",
    "train": "https://raw.githubusercontent.com/amazon-science/mintaka/main/data/mintaka_train.json",
}
POSSIBLE_LANGUAGES = ["en", "ar", "de", "es", "fr", "hi", "it", "ja", "pt"]


def _validate_data_split(split: str):
    if split not in MINTAKA_URI:
        raise ValueError(
            f"MINTAKA only accepts {MINTAKA_URI.keys()} split, but {split} was passed"
        )


def _validate_lang(lang: str):
    if lang not in POSSIBLE_LANGUAGES:
        raise ValueError(
            f"MINTAKA only accepts {POSSIBLE_LANGUAGES}, but {lang} was passed"
        )


def _validate_mode(mode: str):
    if mode not in ["kg", "text"]:
        raise ValueError(f"mode must be kg or text, but {mode} was passed")


def _validate_predictions(
    predictions: Dict[str, Union[str, int, float, List]], mode: str
):
    for key, val in predictions.items():
        if mode == "text" and not isinstance(key, str):
            raise ValueError(
                f"MINTAKA predictions keys must be str, but {key} was passed"
            )
        if mode == "kg" and not (
            val is None or isinstance(val, (str, int, float, list))
        ):
            raise ValueError(
                f"MINTAKA predictions values must be str, int, float, list or None, but {val} was passed"
            )


@MEMORY.cache
def _load_mintaka_data(split: str) -> Dict:
    _validate_data_split(split)

    with urllib.request.urlopen(MINTAKA_URI[split]) as f:
        return ujson.loads(f.read())


def calculate_metrics_for_prediction(
    predictions: Dict[str, Union[str, int, float, List, None]],
    split: str,
    mode: str,
    lang: Optional[str] = "en",
) -> pd.DataFrame:
    """calculate_metrics_for_prediction Helper for calculate DataFrame with predictions and metrics

    Parameters
    ----------
    predictions : Dict[str, Union[str, int, float, List, None]]
        Dict with key - id of question and value - your answer.
        For kg mode it can be entity id, number, string or list of all prevous types
    split : str
        train, validation or test
    mode : str
        'kg' or 'text' for Knowledge Graph or open question text mode
    lang : Optional[str], optional
        Language of used questions,
        can be "en", "ar", "de", "es", "fr", "hi", "it", "ja" or "pt"
        by default "en"

    Returns
    -------
    pd.DataFrame
        DataFrame with at least 6 columns: id, answer, pred, exact_match, f1 and hits1
    """
    _validate_data_split(split)
    _validate_lang(lang)
    _validate_mode(mode)
    _validate_predictions(predictions, mode)

    target_data = _load_mintaka_data(split)
    target_df = pd.DataFrame(target_data)
    target_df["answer"] = target_df["answer"].apply(
        lambda answer: _format_answers(answer, mode, lang)
    )

    predictions_df = pd.DataFrame(predictions.items(), columns=["id", "pred"])
    predictions_df["pred"] = predictions_df["pred"].apply(
        lambda x: _format_predictions(x, mode)
    )

    df = target_df.merge(predictions_df, on="id", how="left")
    df["exact_match"] = df[["pred", "answer"]].apply(
        lambda x: calculate_em(*x, mode), axis=1
    )
    df["f1"] = df[["pred", "answer"]].apply(lambda x: calculate_f1(*x, mode), axis=1)
    df["hits1"] = df[["pred", "answer"]].apply(lambda x: calculate_h1(*x, mode), axis=1)

    return df


def evaluate(
    predictions: Optional[Dict[str, Union[str, int, float, List, None]]] = None,
    df_with_predictions: Optional[pd.DataFrame] = None,
    split: str = "test",
    mode: str = "text",
    lang: Optional[str] = "en",
    groupbycols: List[str] = ["complexityType", "category"],
) -> Dict[str, float]:
    """evaluate - main method for evaluate your predicions

    Parameters
    ----------
    predictions : Optional[Dict[str, Union[str, int, float, List, None]]], optional
        Dict with key - id of question and value - your answer.
        For kg mode it can be entity id, number, string or list of all prevous types,
        by default None
    df_with_predictions : Optional[pd.DataFrame], optional
        Instead of predictions, can be used precomputed dataframe with dataset
        and prediciton that can be calculated in calculate_metrics_for_prediction method,
        by default None
    split : str, optional
        train, validation or test, by default "test"
    mode : str, optional
        'kg' or 'text' for Knowledge Graph or open question text mode,
        by default "text"
    lang : Optional[str], optional
        Language of used questions,
        can be "en", "ar", "de", "es", "fr", "hi", "it", "ja" or "pt"
        by default "en"
    groupbycols : list[str], optional
        List of df_with_predictions columns that was used for grouping results and calculate metrics for different groups.

    Returns
    -------
    Dict[str, float]
        Dict with computed metrics. For all dataset in 'All' field,
        for different splits in corresponding fields: complexityType and category
    """
    if predictions is None and df_with_predictions is None:
        raise ValueError("predictions or df_with_predictions argumet must be provided")
    elif predictions is not None and df_with_predictions is None:
        df_with_predictions = calculate_metrics_for_prediction(
            predictions,
            split,
            mode,
            lang,
        )
    elif predictions is not None and df_with_predictions is not None:
        raise ValueError(
            "predictions and df_with_predictions can not be used at the same call"
        )

    results = {
        "All": {
            "exact_match": df_with_predictions["exact_match"].mean(),
            "exact_match Number Correct Answer Of": (
                df_with_predictions["exact_match"].sum(),
                df_with_predictions["exact_match"].index.size,
            ),
            "f1": df_with_predictions["f1"].mean(),
            "hits1": df_with_predictions["hits1"].mean(),
            "hits1 Number Correct Answer Of": (
                df_with_predictions["hits1"].sum(),
                df_with_predictions["hits1"].index.size,
            ),
        }
    }
    for groupbycol in groupbycols:
        results[groupbycol] = {}
        for group_name, group in df_with_predictions.groupby(groupbycol):
            results_type_metric = {}
            results_type_metric["exact_match"] = group["exact_match"].mean()
            results_type_metric["exact_match Number Correct Answer Of"] = (
                group["exact_match"].sum(),
                group["exact_match"].count(),
            )
            results_type_metric["f1"] = group["f1"].mean()
            results_type_metric["hits1"] = group["hits1"].mean()
            results_type_metric["hits1 Number Correct Answer Of"] = (
                group["hits1"].sum(),
                group["hits1"].count(),
            )

            results[groupbycol][group_name] = results_type_metric

    return results


def _format_answers(answer: dict, mode: str, lang: str) -> Union[list, str, None]:
    """
    Formats answers from the Mintaka test set based on evaluation mode (kg or text)
    Args:
        answer: answer from the Mintaka test set
        mode: mode of evaluation (kg or text)
        lang: language of evaluation (for text answers)
    Returns:
        The answer either as a list for KG evaluation or a string for text evaluation
    """
    if answer["answerType"] == "entity":
        if mode == "kg":
            if answer["answer"] is None:
                return None
            return [ent["name"] for ent in answer["answer"]]  # return a list of Q-codes
        else:
            if answer["answer"] is None:
                return answer[
                    "mention"
                ]  # if no entities linked, return annotator's text answer
            else:
                return " ".join(
                    [
                        ent["label"][lang] if ent["label"][lang] else ent["label"]["en"]
                        for ent in answer["answer"]
                    ]
                )  # return entity labels
    else:
        return str(answer["answer"][0]) if mode == "text" else answer["answer"]


def _format_predictions(pred: object, mode: str) -> Union[list, str, None]:
    """
    Formats predictions to standardized format
    Args:
        pred: predicted answer from a model
        mode: mode of evaluation (kg or text)
    Returns:
        The predicted answer formatted either as a list for KG evaluation or a string for text evaluation
    """
    if pred is None:
        return pred
    elif mode == "text":
        return str(pred)  # return prediction as string
    elif mode == "kg" and not isinstance(pred, list):
        return [pred]  # return prediction as list
    return pred


def normalize_and_tokenize_text(text: str) -> List[str]:
    """
    Normalize and tokenize text based on evaluation script of DPR:
    https://github.com/facebookresearch/DPR/blob/main/dpr/data/qa_validation.py#L175
    Args:
        text: a text answer
    Returns:
        tokens: a list of normalized tokens from the text answer
    """
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"
    _regexp = regex.compile(
        "(%s)|(%s)" % (ALPHA_NUM, NON_WS),
        flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
    )
    text = unicodedata.normalize("NFD", text)
    tokens = []
    matches = [m for m in _regexp.finditer(text)]
    for i in range(len(matches)):
        token = matches[i].group()
        tokens.append(token.lower())
    return tokens


def calculate_em(
    pred: Union[list, str, None], answer: Union[list, str, None], mode: str
) -> int:
    """
    Calculate an exact match score
    Args:
        pred: predicted answer from a model
        answer: answer from the Mintaka test set
        mode: mode of evaluation (kg or text)
    Returns:
        1 if the prediction exactly matches the answer, else 0
    """
    if pd.isnull(pred):
        return False
    if mode == "text" and pred and answer:
        pred = normalize_and_tokenize_text(pred)
        answer = normalize_and_tokenize_text(answer)
        for i in range(0, len(pred) - len(answer) + 1):
            if answer == pred[i : i + len(answer)]:
                return True
        return False
    else:
        return int(pred == answer)


def calculate_f1(pred: Union[str, List], answer: Union[str, List], mode: str) -> float:
    """
    Calculate an F1 score, based on the SQuAD 2.0 evaluate-v2.0.py script
    Args:
        pred: predicted answer from a model
        answer: answer from the Mintaka test set
        mode: mode of evaluation (kg or text)
    Returns:
        An F1 score based on the tokens in a text answer or the list elements in a KG answer
    """
    if pd.isnull(pred):
        return False
    if not answer or not pred:
        return int(answer == pred)
    if mode == "text":
        pred = pred.split()
        answer = answer.split()
    common = collections.Counter(pred) & collections.Counter(answer)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred)
    recall = 1.0 * num_same / len(answer)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calculate_h1(pred: Union[str, List], answer: Union[str, List], mode: str) -> int:
    """
    Calculate a hits@1 score
    Args:
        pred: predicted answer from a model
        answer: answer from the Mintaka test set
        mode: mode of evaluation (kg or text)
    Returns:
        For text or null answers, this is the same as exact match
        For list answers, returns 1 if at least one predicted answer appears in the answer, else 0
    """
    if mode == "text" or pred is None or answer is None:
        return calculate_em(pred, answer, mode)
    else:
        return int(len(collections.Counter(pred) & collections.Counter(answer)) > 0)
