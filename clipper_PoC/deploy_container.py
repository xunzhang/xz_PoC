import os
import sys
sys.path.insert(0, '/Users/xunzhang/Desktop/2018/github/clipper/clipper_admin/')
import tensorflow as tf
from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.tensorflow import deploy_tensorflow_model

cur_dir = os.path.dirname(os.path.abspath(__file__))

app_name = "tf-lr-app"
model_name = "tf-lr-model"

if __name__ == "__main__":
    clipper_conn = ClipperConnection(DockerContainerManager())
    clipper_conn.stop_all()
    clipper_conn.start_clipper()
    clipper_conn.register_application(name=app_name,
                                      input_type="integers",
                                      default_output="rabbit",
                                      slo_micros=100000)
    print(os.path.abspath("data"))
    clipper_conn.build_and_deploy_model(name=model_name,
                                        version=1,
                                        input_type="integers",
                                        model_data_path=os.path.abspath("data"),
                                        base_image="xunzhang/tf_lr_container:latest",
                                        num_replicas=1)
    clipper_conn.link_model_to_app(app_name, model_name)
    print(clipper_conn.get_clipper_logs())
