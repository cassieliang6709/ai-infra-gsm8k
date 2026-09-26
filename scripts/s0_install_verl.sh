#!/bin/bash
# Install verl in its own venv (inherits system torch 2.13 / vLLM 0.29):
# verl pins transformers==5.12.1 while ms-swift needs 5.16.1, so they cannot share one env.
set -eo pipefail
cd /root/autodl-tmp
[ -d envverl ] || /root/miniconda3/bin/python -m venv --system-site-packages envverl
source envverl/bin/activate
pip install -q "transformers==5.12.1" "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
  "TransferQueue>=0.1.10" pyzmq msgspec codetiming hydra-core pylatexenc "ray[default]>=2.41.0" \
  torchdata mathruler qwen_vl_utils cachetools nvtx math_verify latex2sympy2_extended
pip install -q --no-deps -e /root/autodl-tmp/verl
python -c "import verl, torch, vllm, transformers; print('verl', verl.__version__, 'torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
echo "S0 INSTALL OK"
