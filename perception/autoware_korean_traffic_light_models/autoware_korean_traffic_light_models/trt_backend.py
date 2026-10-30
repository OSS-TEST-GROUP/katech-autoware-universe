"""Small synchronous TensorRT 10 runtime; device buffers have explicit ownership."""
import ctypes as C
import hashlib
import json
from pathlib import Path
import numpy as np
import tensorrt_bindings as trt


class TRTBackend:
    def __init__(self, engine_path, require_validated=False):
        engine_path=Path(engine_path)
        if require_validated:
            report=json.loads((engine_path.parent/'trt_validation.json').read_text())
            matches=[r for r in report['results'] if r['engine']==engine_path.name]
            if len(matches)!=1 or not matches[0]['passed']:
                raise ValueError('Engine has not passed numerical validation')
            if hashlib.sha256(engine_path.read_bytes()).hexdigest()!=matches[0]['engine_sha256']:
                raise ValueError('Engine hash differs from validated artifact')
        self.log = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.log)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError("Cannot deserialize TensorRT engine")
        assert self.engine.num_io_tensors == 2
        self.input_name = next(self.engine.get_tensor_name(i) for i in range(2)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT)
        self.output_name = next(self.engine.get_tensor_name(i) for i in range(2)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT)
        self.shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.batch = self.shape[0]
        assert self.shape == (self.batch, 3, 20, 70) and self.batch in (1, 6)
        assert tuple(self.engine.get_tensor_shape(self.output_name)) == (self.batch, 6)
        assert self.engine.get_tensor_dtype(self.input_name) == trt.float32
        assert self.engine.get_tensor_dtype(self.output_name) == trt.float32
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Cannot create TensorRT execution context")
        self.cuda = C.CDLL("libcudart.so.12")
        signatures = {
            "cudaMalloc": [C.POINTER(C.c_void_p), C.c_size_t],
            "cudaFree": [C.c_void_p],
            "cudaMemcpy": [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int],
            "cudaStreamCreate": [C.POINTER(C.c_void_p)],
            "cudaStreamDestroy": [C.c_void_p],
            "cudaStreamSynchronize": [C.c_void_p],
        }
        for name, signature in signatures.items():
            f = getattr(self.cuda, name)
            f.argtypes = signature
            f.restype = C.c_int
        self.cuda.cudaGetErrorString.argtypes = [C.c_int]
        self.cuda.cudaGetErrorString.restype = C.c_char_p
        self.stream, self.device_in, self.device_out = C.c_void_p(), C.c_void_p(), C.c_void_p()
        self.output = np.empty((self.batch, 6), np.float32)
        try:
            self.check(self.cuda.cudaStreamCreate(C.byref(self.stream)))
            self.check(self.cuda.cudaMalloc(C.byref(self.device_in), int(np.prod(self.shape)) * 4))
            self.check(self.cuda.cudaMalloc(C.byref(self.device_out), self.output.nbytes))
            assert self.context.set_tensor_address(self.input_name, self.device_in.value)
            assert self.context.set_tensor_address(self.output_name, self.device_out.value)
        except Exception:
            self.close()
            raise

    def check(self, status):
        if status:
            raise RuntimeError(self.cuda.cudaGetErrorString(status).decode())

    def infer(self, data):
        if data.shape != self.shape or data.dtype != np.float32:
            raise ValueError(f"Expected float32 {self.shape}, got {data.dtype} {data.shape}")
        if not np.isfinite(data).all() or np.any(data < 0) or np.any(data > 1):
            raise ValueError("Input outside finite [0,1]")
        data = np.ascontiguousarray(data)
        self.check(self.cuda.cudaMemcpy(self.device_in, data.ctypes.data, data.nbytes, 1))
        if not self.context.execute_async_v3(self.stream.value):
            raise RuntimeError("TensorRT inference failed")
        self.check(self.cuda.cudaStreamSynchronize(self.stream))
        self.check(self.cuda.cudaMemcpy(self.output.ctypes.data, self.device_out, self.output.nbytes, 2))
        return self.output.copy()

    def close(self):
        if getattr(self, "stream", None) and self.stream.value:
            self.cuda.cudaStreamSynchronize(self.stream)
        for name in ("device_in", "device_out"):
            pointer = getattr(self, name, None)
            if pointer is not None and pointer.value:
                self.cuda.cudaFree(pointer)
                pointer.value = None
        pointer = getattr(self, "stream", None)
        if pointer is not None and pointer.value:
            self.cuda.cudaStreamDestroy(pointer)
            pointer.value = None
        self.context = None
        self.engine = None
        self.runtime = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

