#!/bin/bash
# EMR bootstrap action, on every node before Spark starts (AWS validation): Python 3.11 with the
# local env's package versions (EMR 7.5's own Python 3.9 is too old for torch 2.10) and the code, unpacked to
# /opt/lakegrl/lakegrl-code (its `pipeline` symlink points at itself, so the repo root on PYTHONPATH is enough).
#   Path=s3://us-east-1-s3-gnn/LakeGRL/code/bootstrap.sh,Args=[s3://us-east-1-s3-gnn/LakeGRL/code/lakegrl-code.tar.gz]
set -euxo pipefail
sudo dnf install -y python3.11 python3.11-pip
PIP="sudo python3.11 -m pip install --no-cache-dir"
$PIP torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
$PIP torch-geometric==2.8.1 numpy==1.26.4 pandas==2.3.3 pyarrow==17.0.0 scikit-learn==1.9.1 scipy==1.17.1 \
     psutil==7.2.2 boto3
$PIP --no-deps delta-spark==3.2.0   # its Python module only: EMR's delta-defaults puts the Delta jars on the classpath
sudo mkdir -p /opt/lakegrl/lakegrl-code
aws s3 cp "${1:?code tarball on S3}" /tmp/lakegrl-code.tar.gz
sudo tar -xzf /tmp/lakegrl-code.tar.gz -C /opt/lakegrl/lakegrl-code
sudo chmod -R a+rX /opt/lakegrl
sudo chown -R hadoop:hadoop /opt/lakegrl
python3.11 -c "import torch, torch_geometric, pyarrow; print('lakegrl bootstrap ok', torch.__version__, pyarrow.__version__)"
