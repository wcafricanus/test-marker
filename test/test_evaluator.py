import json
import os
import pytest

from prediction.Preprocess import strip_label, draw_greyscale_digit, preprocess_data
from prediction.evaluator import Evaluator


@pytest.fixture
def evaluator():
    e = Evaluator()
    return e


@pytest.fixture
def test_dict():
    cwd = os.path.dirname(__file__)
    with open(cwd+'/test2.txt', 'r') as myfile:
        data = json.load(myfile)
    data = data['test']['vasCogBlock']
    return preprocess_data(data)


def test_evaluator(evaluator, test_dict):
    result = evaluator.evaluate(test_dict)
