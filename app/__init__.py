from flask import Flask

from prediction.evaluator import Evaluator

app = Flask(__name__)
app.evaluator = Evaluator()

from app import apiserver