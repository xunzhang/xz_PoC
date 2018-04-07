import os
import sys
sys.path.insert(0, '/Users/xunzhang/Desktop/2018/github/clipper/clipper_admin/')
import tensorflow as tf
from clipper_admin import ClipperConnection, DockerContainerManager
from clipper_admin.deployers.tensorflow import deploy_tensorflow_model

cur_dir = os.path.dirname(os.path.abspath(__file__))

app_name = "tf-lr-app"
model_name = "tf-lr-model"

CHECKPOINT_PATH="data/model.ckpt"
FROZEN_GRAPH_PATH="frozen_graph/export_dir"


def predict(sess, inputs):
    preds = sess.run('predict_class:0', feed_dict={'pixels:0': inputs})
    return [str(p) for p in preds]

def load_from_ckp():
    pass

def load_from_frozen():
    pass

if __name__ == "__main__":
    clipper_conn = ClipperConnection(DockerContainerManager())
    clipper_conn.stop_all()
    clipper_conn.start_clipper()
    clipper_conn.register_application(name=app_name,
                                      input_type="integers",
                                      default_output="rabbit",
                                      slo_micros=100000)
    if len(sys.argv) != 3:
        print("invalid usage")
        print("usage: python deploy.py --mode checkpoint|frozen|sess-checkpoint|sess-frozen")
        exit(1)
    if sys.argv[2] == "sess-checkpoint":
        sess = load_from_ckp()
        deploy_tensorflow_model(clipper_conn,
                                model_name,
                                version=1,
                                input_type="integers",
                                func=predict, tf_sess_or_saved_model_path=sess)
    elif sys.argv[2] == "sess-frozen":
        sess = load_from_frozen()
        deploy_tensorflow_model(clipper_conn,
                                model_name,
                                version=1,
                                input_type="integers",
                                func=predict, tf_sess_or_saved_model_path=sess)
    elif sys.argv[2] == "checkpoint":
        deploy_tensorflow_model(clipper_conn,
                                model_name,
                                version=1,
                                input_type="integers",
                                func=predict, tf_sess_or_saved_model_path=CHECKPOINT_PATH.split('/')[0])
    elif sys.argv[2] == "frozen":
        deploy_tensorflow_model(clipper_conn,
                                model_name,
                                version=1,
                                input_type="integers",
                                func=predict, tf_sess_or_saved_model_path=FROZEN_GRAPH_PATH)
    else:
        print("invalid usage")
        print("usage: python deploy.py --mode checkpoint|frozen|checkpoint|sess-frozen")
        exit(1)
    clipper_conn.link_model_to_app(app_name, model_name)
