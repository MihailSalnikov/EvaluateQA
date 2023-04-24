[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# EvaluateQA

Package for evaluate QA datasets and Leaderboard with SOTA approaches

## Install

```bash
pip install evaluateqa
```

## Supported datasets


### [Mintaka: A Complex, Natural, and Multilingual Dataset for End-to-End Question Answering](https://github.com/amazon-science/mintaka)

```python
from evaluateqa.mintaka import evaluate

predictions = {
    '9ace9041': 'Q90',
    '9ace9042': 3,
    ...
}

results = evaluate(
    predictions,
    split='test',
    mode='kg',
    lang='en',
)
```



