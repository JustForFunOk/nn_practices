import onnxruntime as ort
import numpy as np
import sys


if __name__ == '__main__':
    onnx_model = sys.argv[1]

    ort_session = ort.InferenceSession(
        onnx_model, providers=['CPUExecutionProvider'])

    input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)

    output = ort_session.run(['output'], {'input': input_data})

    print(output)
