import json
import os
import time
import uuid
from flask import Flask, send_file
from flask import request
from flask_cors import CORS

from processing import generate_floor_plan


def design_all():

    files = os.listdir(r'.\inputs')
    files = [os.path.join(r'.\inputs', file) for file in files if file.endswith('.json')]
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
        name = file.split('.')[0]
        res = generate_floor_plan(data, output_name=name)

design_all()