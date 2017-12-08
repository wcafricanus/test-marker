import json
import os
import pytest

from prediction.Preprocess import strip_label, draw_greyscale_digit, preprocess_data, preprocess_single_data
from prediction.evaluator_tf import Evaluator


@pytest.fixture
def evaluator():
    e = Evaluator()
    return e


@pytest.fixture
def test_dict():
    cwd = os.path.dirname(__file__)
    with open(cwd+'/test3.txt', 'r') as myfile:
        data = json.load(myfile)
    vas_cog_block = data['test']['vasCogBlock']
    vas_block_size = data['test']['vasBlockSize']
    return preprocess_data(vas_cog_block, vas_block_size)


@pytest.fixture
def single_item_dict():
    cwd = os.path.dirname(__file__)
    with open(cwd + '/mark_one_body.txt', 'r') as myfile:
        data = json.load(myfile)
    vas_cog_block = data['vasCogBlock']
    vas_block_size = data['vasBlockSize']
    return preprocess_single_data(vas_cog_block, vas_block_size)


def test_evaluator(evaluator, test_dict, single_item_dict):
    result = evaluator.evaluate(test_dict)
    result_single = evaluator.evaluate(single_item_dict)



