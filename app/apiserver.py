from flask import request, jsonify

from app import app
from prediction.Preprocess import preprocess_data, preprocess_single_data


@app.route('/mark', methods=['POST'])
def mark():
    raw_data = request.get_json()
    vas_cog_block = raw_data['vas_cog_block']
    vas_block_size = raw_data['vas_block_size']
    data = preprocess_data(vas_cog_block, vas_block_size)
    result_dict = app.evaluator.evaluate(data)
    return jsonify(result_dict)


@app.route('/mark_one', methods=['POST'])
def mark_one():
    raw_data = request.get_json()
    vas_cog_block = raw_data['vas_cog_block']
    vas_block_size = raw_data['vas_block_size']
    data = preprocess_single_data(vas_cog_block, vas_block_size)
    result_dict = app.evaluator.evaluate(data)
    return jsonify(result_dict)
