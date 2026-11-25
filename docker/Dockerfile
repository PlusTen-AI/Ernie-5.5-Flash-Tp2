# ernie55-plus image — builds and runs the C++/CUDA engine and its server on GB10 (sm_121a).
# Base (local, nothing pulled): NGC PyTorch 26.07 = CUDA 13.3 · cmake 3.31 · ninja · gcc 13 · NCCL · ibverbs/mlx5 ·
# nsys/ncu · torch/numpy/safetensors (the reference forward of the correctness gate) · CuTe DSL.
# Tag = ernie55-plus:<first 8 of this file's sha256>; a changed file is a new tag, old tags are never overwritten.
FROM nvcr.io/nvidia/pytorch:26.07-py3
LABEL project=ernie-5.5-flash
ENV CUDAARCHS=121a \
    TORCH_CUDA_ARCH_LIST=12.1a \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# server front: HTTP only, pinned, no dependency resolution — tokenizers, jinja2, pydantic, anyio, click and h11 come
# with the base image
RUN pip install --no-deps fastapi==0.115.6 starlette==0.41.3 uvicorn==0.32.1 && \
    python -c "import fastapi, starlette, uvicorn, tokenizers, jinja2, safetensors"
# no NGC banner in docker logs: plain entrypoint
ENTRYPOINT []
WORKDIR /work/src
CMD ["bash"]
