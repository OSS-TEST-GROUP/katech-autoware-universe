"""Pinned candidate-C CPU backend for evaluation, not a real-time guarantee."""
import numpy as np
from .model_io import model_record


class ONNXBackend:
    def __init__(self, path):
        row = model_record(path, 'classifier')
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.batch = row['batch']
        self.shape = (self.batch, 3, 20, 70)
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or tuple(inputs[0].shape) != self.shape or tuple(outputs[0].shape) != (self.batch, 6):
            raise ValueError('Candidate C tensor shape mismatch')
        if inputs[0].type != 'tensor(float)' or outputs[0].type != 'tensor(float)':
            raise ValueError('Candidate C requires float32')
        self.input_name = inputs[0].name

    def infer(self, data):
        if data.shape != self.shape or data.dtype != np.float32 or not np.isfinite(data).all() or np.any(data < 0) or np.any(data > 1):
            raise ValueError('Invalid candidate C input')
        return self.session.run(None, {self.input_name: np.ascontiguousarray(data)})[0]

    def close(self):
        self.session = None
