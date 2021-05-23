import os



filt_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(filt_path) + os.path.sep + ".")

# crnn参数
dbnet_model_path = os.path.join(father_path, "models/dbnet.onnx")
crnn_model_path = os.path.join(father_path, "models/crnn_lite_lstm.onnx")


# angle
angle_net_path = os.path.join(father_path, "models/angle_net.onnx")

