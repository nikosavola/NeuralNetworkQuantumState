# Neural Network Quantum State

Group work for [PHYS-E0421 - Solid State Physics](https://mycourses.aalto.fi/course/view.php?id=31530) course at Aalto University


## Installation

### Linux and macOS

Install dependencies with:
```bash
pip install -r requirements.txt
```
For optional GPU support, run additionally
```bash
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

### Windows

Not recommended natively. Use WSL instead and follow the Linux approach. If you really want you can install w/o `mpi` with an unofficial `jaxlib`:
```bash
pip install jaxlib -f https://whls.blob.core.windows.net/unstable/index.html
pip install -r requirements.txt
```
