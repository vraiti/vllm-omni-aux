# Installing vLLM-Omni

## 1. Create a UV venv

```bash
uv venv ~/vraiti/vpp/.venv --python 3.12
source ~/vraiti/vpp/.venv/bin/activate
```

## 2. Determine the required vLLM version

Check the base image in `docker/Dockerfile.cuda`:

```bash
head -1 vllm-omni/docker/Dockerfile.cuda
```

The `BASE_IMAGE` tag contains the vLLM version. For example:

```
ARG BASE_IMAGE=vllm/vllm-openai:v0.24.0
```

means vLLM v0.24.0 is required.

## 3. Install vLLM

```bash
uv pip install vllm==0.24.0
```

Replace `0.24.0` with whatever version the Dockerfile specifies.

## 4. Clone and install vLLM-Omni

Clone the branch you need:

```bash
cd ~/vraiti/vpp
git clone --branch <branch> --single-branch git@github.com:vraiti/vllm-omni.git
```

Install in editable mode with dev dependencies:

```bash
cd vllm-omni
uv pip install -e '.[dev]'
```

## 5. Verify

```bash
python3 -c "import vllm; import vllm_omni; print('vllm', vllm.__version__); print('vllm_omni', vllm_omni.__version__)"
```
