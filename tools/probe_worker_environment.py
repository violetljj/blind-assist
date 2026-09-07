"""Small runtime checks using generated data only; no research cohorts/models."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['research', 'l10', 'export'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'profile': args.profile, 'python': sys.version, 'evidence': 'generated runtime smoke only'}
    if args.profile == 'research':
        import cv2
        import imageio_ffmpeg
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        import torchvision
        from transformers import BertConfig, BertModel
        assert torch.cuda.is_available()
        boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]], device='cuda')
        assert torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device='cuda'), 0.5).numel() == 1
        model = BertModel(BertConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64)).cuda().eval()
        with torch.inference_mode():
            assert tuple(model(torch.zeros((1, 4), dtype=torch.long, device='cuda')).last_hidden_state.shape) == (1, 4, 32)
        image = np.zeros((8, 8, 3), np.uint8)
        ok, png = cv2.imencode('.png', image)
        assert ok and cv2.imdecode(png, cv2.IMREAD_COLOR).shape == image.shape
        ort.preload_dlls()
        graph = onnx.helper.make_graph([onnx.helper.make_node('Add', ['x', 'x'], ['y'])], 'gpu-add',
            [onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [4])],
            [onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [4])])
        proto = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 17)], ir_version=8)
        options = ort.SessionOptions()
        options.enable_profiling = True
        options.profile_file_prefix = str(args.output / 'onnx-profile')
        options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
        session = ort.InferenceSession(proto.SerializeToString(), sess_options=options, providers=['CUDAExecutionProvider'])
        actual = session.run(None, {'x': np.arange(4, dtype=np.float32)})[0]
        assert np.array_equal(actual, np.arange(4, dtype=np.float32) * 2)
        events = json.loads(Path(session.end_profiling()).read_text())
        providers = {e.get('args', {}).get('provider') for e in events if e.get('args', {}).get('provider')}
        assert 'CUDAExecutionProvider' in providers, providers
        import subprocess
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=black:s=32x32:d=0.1', '-y', str(args.output / 'video.mp4')], check=True)
        result.update(torch=torch.__version__, torchvision=torchvision.__version__, onnxruntime=ort.__version__, actualProviders=sorted(providers), transformer='tiny random BERT CUDA forward passed', ffmpeg=ffmpeg)
    elif args.profile == 'l10':
        import numpy as np
        import pandas as pd
        import tables
        import networkx as nx
        frame = pd.DataFrame({'value': np.arange(4)})
        target = args.output / 'table.h5'
        frame.to_hdf(target, key='test')
        assert frame.equals(pd.read_hdf(target, key='test'))
        assert nx.shortest_path(nx.path_graph(3), 0, 2) == [0, 1, 2]
        result.update(numpy=np.__version__, pandas=pd.__version__, tables=tables.__version__)
    else:
        import numpy as np
        import tensorflow as tf
        import onnx2tf
        import onnx_graphsurgeon
        import onnxruntime
        import ultralytics
        from ai_edge_litert.interpreter import Interpreter
        model = tf.keras.Sequential([tf.keras.layers.Input(shape=(2,)), tf.keras.layers.Dense(1, use_bias=False, kernel_initializer='ones')])
        payload = tf.lite.TFLiteConverter.from_keras_model(model).convert()
        (args.output / 'tiny.tflite').write_bytes(payload)
        runtime = Interpreter(model_content=payload)
        runtime.allocate_tensors()
        runtime.set_tensor(runtime.get_input_details()[0]['index'], np.array([[1, 2]], dtype=np.float32))
        runtime.invoke()
        assert np.allclose(runtime.get_tensor(runtime.get_output_details()[0]['index']), [[3.]])
        result.update(tensorflow=tf.__version__, conversion='Keras to TFLite with real interpreter invocation passed', backend='CPU export/conversion')
    result['status'] = 'PASS'
    (args.output / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
