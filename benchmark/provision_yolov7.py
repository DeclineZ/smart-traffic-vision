"""Build YOLOv7 engines with decoded xywh and objectness-weighted class scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'yolov7'))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['yolov7', 'yolov7x'], required=True)
    p.add_argument('--download-only', action='store_true', help='Download weights without exporting or using GPU')
    a = p.parse_args()
    folder = ROOT / 'models'
    weights = folder / (a.model + '.pt')
    url = f'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{a.model}.pt'
    if not weights.exists():
        partial = weights.with_suffix('.download')
        urllib.request.urlretrieve(url, partial)
        partial.replace(weights)
    if a.download_only:
        print(weights, weights.stat().st_size, digest(weights), flush=True)
        return
    import torch
    import tensorrt as trt
    target = folder / (a.model + '.engine')
    if target.exists():
        raise FileExistsError(f'Preserving existing engine: {target}')
    # Official checkpoints contain module objects; load only the trusted source above.
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    model = (ckpt.get('ema') or ckpt['model']).float().eval()
    params = sum(p.numel() for p in model.parameters())
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    model = model.fuse().eval().cuda()
    for m in model.modules():
        if isinstance(m, torch.nn.Upsample):
            m.recompute_scale_factor = None
    model.model[-1].export = False
    model.model[-1].concat = True

    class Scores(torch.nn.Module):
        def __init__(self, network):
            super().__init__()
            self.network = network

        def forward(self, image):
            y = self.network(image)
            if isinstance(y, tuple):
                y = y[0]
            return torch.cat((y[..., :4], y[..., 5:] * y[..., 4:5]), dim=-1).transpose(1, 2)

    wrapped = Scores(model).eval()
    image = torch.zeros(1, 3, 640, 640, device='cuda')
    with torch.no_grad():
        wrapped(image)
        onnx = weights.with_suffix('.onnx')
        torch.onnx.export(wrapped, image, str(onnx), opset_version=17, dynamo=False,
                          input_names=['images'], output_names=['output0'])
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    start = time.perf_counter()
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError('TensorRT build failed')
    metadata = dict(task='detect', imgsz=[640, 640], batch=1, stride=32, names=names, parameters=params,
                    end2end=False, args=dict(quantize=16, nms=False, dynamic=False),
                    description='YOLOv7 decoded xywh + objectness*class scores, external NMS')
    header = json.dumps(metadata).encode()
    with target.open('xb') as f:
        f.write(len(header).to_bytes(4, 'little'))
        f.write(header)
        f.write(bytes(engine))
    manifest = dict(model=a.model, source_url=url, weights_sha256=digest(weights),
                    engine_sha256=digest(target), parameters=params, precision='fp16',
                    input_size=640, batch=1, tensorrt=trt.__version__, torch=torch.__version__,
                    cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                    build_seconds=time.perf_counter()-start, output_schema=metadata['description'])
    target.with_suffix('.build.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('BUILT', target, flush=True)


if __name__ == '__main__':
    main()
