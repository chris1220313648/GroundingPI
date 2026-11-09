#!/usr/bin/env bash
# Reuse the validated Torch/GroundingPI runtime read-only; install additions locally.
set -euo pipefail
base="${BASE_ENV:-/share/project/robot-rfm/robot-exp/usr/cjy/envs/starvla}"
export OPENWAM_ENV="${OPENWAM_ENV:-/share/project/robot-rfm/robot-exp/usr/cjy/envs/openwam-groundingpi}"
root="$(cd "$(dirname "$0")/.." && pwd)"
"$base/bin/python" -m venv --system-site-packages "$OPENWAM_ENV"
"$OPENWAM_ENV/bin/python" - "$base" "$root" <<'PY'
import sys,sysconfig
from pathlib import Path
base=Path(sys.argv[1]); site=Path(sysconfig.get_path('purelib'))
parent=base/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
(site/'starvla_runtime.pth').write_text(str(parent)+'\n'+sys.argv[2]+'\n')
for name in ['__editable__.transformers-5.7.0.pth','distutils-precedence.pth']:
    (site/('zz-'+name)).write_text((parent/name).read_text())
shim=Path(sys.executable).parent/'torchrun'
shim.write_text('#!'+sys.executable+'\nfrom torch.distributed.run import main\nmain()\n')
shim.chmod(0o755)
PY
"$OPENWAM_ENV/bin/python" -m pip install numpy==1.26.4 matplotlib==3.10.8 accelerate==1.10.1 hydra-core==1.3.8 h5py==3.16.0 decord==0.6.0 lmdb==3.0.0
