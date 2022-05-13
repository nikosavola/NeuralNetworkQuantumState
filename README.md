# Neural Network Quantum State

Group work for [PHYS-E0421 - Solid State Physics](https://mycourses.aalto.fi/course/view.php?id=31530) course at Aalto University.

Check [`notebook.ipynb`](https://github.com/nikosavola/NeuralNetworkQuantumState/blob/main/notebook.ipynb) for running the model and results. Source code for the problem, model, and Ray Tune compability is in [`model.py`](https://github.com/nikosavola/NeuralNetworkQuantumState/blob/main/model.py).

## Installation

### Linux and macOS

Install dependencies with:
```bash
pip install -r requirements.txt
```
For optional GPU support, install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-download). Afterwards run:
```bash
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```
For optional but recommended MPI support for CPU and GPU, install netket with MPI deps.:
```bash
pip install --upgrade "netket[mpi]"
```
With GPU, this requires a correctly setup CUDA compiler, so you might have to do something like `export CUDA_PATH=/usr/local/cuda/`.

### Windows

Not recommended natively. Use WSL instead and follow the Linux approach. If you really want you can install w/o `mpi` with an unofficial `jaxlib`:
```bash
pip install jaxlib -f https://whls.blob.core.windows.net/unstable/index.html
pip install -r requirements.txt
```
