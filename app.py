from flask import Flask, request, jsonify
import requests
import json
from datetime import datetime
from datetime import timedelta
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

@app.route("/", methods=["POST"])
def register():
    data = request.get_json()
    return jsonify(data)