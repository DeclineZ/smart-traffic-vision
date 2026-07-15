import torch
import numpy as np
import tensorrt as trt
from .tools import get_logger

INPUT_NAME = "images"
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

logger = get_logger("TRTModel")

class TRTModel:
    def __init__(self, engine_path: str, input_shape: tuple, device: torch.device):
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.device = device
        self.use_new_api = False  # Flag to track TensorRT API version

        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda:0")

        torch.cuda.set_device(self.device)
        self.trt_stream = torch.cuda.Stream(device=self.device)

        logger.info("Loading TensorRT engine")
        self.engine = self.load_engine()
        self.context = self.engine.create_execution_context()

        self.input_tensor = torch.empty(
            self.input_shape,
            device=self.device,
            dtype=torch.float32,
        )
        self.outputs = {}

        # Detect which TensorRT API is available
        if hasattr(self.context, 'set_tensor_address'):
            self.use_new_api = True
            # Only set input shape if the engine has dynamic shapes
            if hasattr(self.context, 'set_input_shape'):
                try:
                    self.context.set_input_shape(INPUT_NAME, self.input_shape)
                except Exception as e:
                    logger.warning(f"Could not set input shape (engine may have fixed shapes): {e}")
            
            self.context.set_tensor_address(INPUT_NAME, int(self.input_tensor.data_ptr()))
            logger.info(f"Allocated input tensor '{INPUT_NAME}' | shape={self.input_shape} | dtype={self.input_tensor.dtype}")
            self.allocate_buffers()
        else:
            # Use older TensorRT API (pre-10.0)
            logger.info("Using older TensorRT API (pre-10.0)")
            self.allocate_buffers_legacy()
        
        logger.info("Running TensorRT warmup")
        self.warmup()

    def load_engine(self):
        with open(self.engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        logger.info("TensorRT engine loaded successfully")
        return engine

    def allocate_buffers(self):
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))

                tensor = torch.empty(
                    shape,
                    dtype=torch.from_numpy(np.empty((), dtype=dtype)).dtype,
                    device=self.device,
                )

                self.outputs[name] = tensor
                self.context.set_tensor_address(name, int(tensor.data_ptr()))

                logger.info(
                    f"Allocated output tensor '{name}' | shape={shape} | dtype={tensor.dtype}"
                )

    def allocate_buffers_legacy(self):
        """Allocate buffers using older TensorRT API (pre-10.0)"""
        self.bindings = []
        self.binding_addrs = []
        
        # Find input binding index
        input_idx = None
        for i in range(self.engine.num_bindings):
            if self.engine.binding_is_input(i):
                input_idx = i
                break
        
        if input_idx is None:
            raise RuntimeError("Could not find input binding in engine")
        
        # Set input binding
        self.bindings.append(int(self.input_tensor.data_ptr()))
        
        # Set output bindings
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                name = self.engine.get_binding_name(i)
                shape = tuple(self.context.get_binding_shape(i))
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                
                tensor = torch.empty(
                    shape,
                    dtype=torch.from_numpy(np.empty((), dtype=dtype)).dtype,
                    device=self.device,
                )
                
                self.outputs[name] = tensor
                self.bindings.append(int(tensor.data_ptr()))
                
                logger.info(
                    f"Allocated output tensor '{name}' | shape={shape} | dtype={tensor.dtype}"
                )

    def warmup(self, iters: int = 30):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        times = []

        for _ in range(iters):
            self.input_tensor.normal_()

            starter.record(stream=self.trt_stream)
            with torch.cuda.stream(self.trt_stream):
                if self.use_new_api:
                    self.context.execute_async_v3(self.trt_stream.cuda_stream)
                else:
                    self.context.execute_async_v2(self.bindings, self.trt_stream.cuda_stream)
            ender.record(stream=self.trt_stream)

            self.trt_stream.synchronize()
            times.append(starter.elapsed_time(ender) / 1000.0)

        logger.info(f"Warmup complete | Avg inference time: {sum(times)/len(times):.4f} s")

    def infer(self, inp: torch.Tensor) -> dict:
        if inp.device.type != self.device.type or inp.device.index != self.device.index:
            raise ValueError("Input tensor must be on the same CUDA device as the engine")

        if inp.shape != self.input_shape:
            raise ValueError(f"Expected input shape {self.input_shape}, got {inp.shape}")

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        with torch.cuda.stream(self.trt_stream):
            self.input_tensor.copy_(inp, non_blocking=True)
            starter.record(stream=self.trt_stream)
            if self.use_new_api:
                self.context.execute_async_v3(self.trt_stream.cuda_stream)
            else:
                self.context.execute_async_v2(self.bindings, self.trt_stream.cuda_stream)
            ender.record(stream=self.trt_stream)

        self.trt_stream.synchronize()
        infer_time = starter.elapsed_time(ender) / 1000.0

        return infer_time, self.outputs