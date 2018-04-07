import sys
sys.path.insert(0, '/Users/xunzhang/Desktop/2018/github/clipper/clipper_admin/')
import time
import json
import requests
import numpy as np
from clipper_admin import ClipperConnection, DockerContainerManager

app_name = "tf-lr-app"
headers = {'Content-type': 'application/json'}

def get_test_point():
    return [np.random.randint(255) for _ in range(784)]

def test_model(clipper_conn, app, version):
    time.sleep(25)
    num_preds = 25
    num_defaults = 0
    addr = clipper_conn.get_query_addr()
    print(addr)
    for i in range(num_preds):
        response = requests.post(
            "http://%s/%s/predict" % (addr, app),
            headers=headers,
            data=json.dumps({
                'input': get_test_point()
            }))
        result = response.json()
        print(result)
        if response.status_code == requests.codes.ok and result["default"]:
            num_defaults += 1
        elif response.status_code != requests.codes.ok:
            print(result)
            raise BenchmarkException(response.text)

    if num_defaults > 0:
        print("Error: %d/%d predictions were default" % (num_defaults,
                                                         num_preds))

if __name__ == "__main__":
    clipper_conn = ClipperConnection(DockerContainerManager())
    clipper_conn.connect()
    test_model(clipper_conn, app_name, 1)
