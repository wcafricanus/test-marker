from flask import request, jsonify

from app import app
from prediction.Preprocess import preprocess_data


@app.route('/mark', methods=['POST'])
def mark():
    raw_data = request.get_json()
    data = preprocess_data(raw_data)
    result_dict = app.evaluator.evaluate(data)
    return jsonify(result_dict)
