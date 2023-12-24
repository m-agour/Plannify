import os
import time
import uuid
from processing import generate_floor_plan
from flask import Flask, send_file
from flask import request
from flask_cors import CORS


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
app = Flask(__name__)

app.config['SECRET_KEY'] = 'hello darkness my old friend'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS()
cors.init_app(app)


@app.route('/', methods=['GET'])
def request_test():
    if request.method == 'GET':
        return 'AI service is up and running. Say thank you demon!'
    else:
        return "POST Error 405 Method Not Allowed"


@app.route('/design',
           methods=['POST', 'GET', 'PUT', 'DELETE', 'PATCH', 'OPTIONS',
                    'HEAD'])
def design():
    if request.method == 'POST':
        data = request.get_json()
        # generate date dependent name
        name = time.strftime("%Y%m%d-%H%M%S")
        name = "output"
        res = generate_floor_plan(data, output_name=name)
        # sleep(5)
        return "res"
        # return "POST request received. Thank you demon!"

    else:
        return (f"{request.method} requests are not allowed at this endpoint. "
                f"Only POST requests are allowed.")


@app.route('/download_model')
def download_file():
    random_name = str(uuid.uuid4()) + '.gltf'
    return send_file(r'.\outputs\gltf\output.gltf',
                     download_name=random_name)


@app.route('/download_floor_plan')
def download_floor_plan():
    random_name = str(uuid.uuid4()) + '.png'
    return send_file(r'.\outputs\images\output.png',
                     download_name=random_name)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
