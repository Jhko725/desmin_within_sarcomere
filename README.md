# Desmin filaments extend along actin filaments within the mouse skeletal muscle sarcomere

[![DOI](https://zenodo.org/badge/1392397862.svg)](https://zenodo.org/badge/latestdoi/1392397862)

This repository contains code accompanying the study described below.

**Publication status:** Accepted for publication in *Nature Communications* (2026).

## Authors

Sunu Joseph<sup>1,†</sup>, Joon-Hyuk Ko<sup>2,3,†</sup>, Nitsan Dahan<sup>4</sup>, Tali Dadosh<sup>5</sup>, Hana K. Pak<sup>6</sup>, Jonathan A. Kirk<sup>6</sup>, Wonho Jhe<sup>2,7,\*</sup>, and Shenhav Shemer<sup>1,\*</sup>

<sup>†</sup> These authors contributed equally.

## Affiliations

1. Faculty of Biology, Technion Institute of Technology, Haifa, Israel.
2. Center for 0D Nanofluidics, Department of Physics and Astronomy & Institute of Applied Physics, Seoul National University, Korea.
3. Center for AI and Natural Sciences, Korea Institute for Advanced Study, Seoul, Korea.
4. Lorry I. Lokey Interdisciplinary Center, Life Sciences and Engineering Infrastructure Center, Technion Institute of Technology, Haifa, Israel.
5. Department of Chemical Research Support, Weizmann Institute of Science, Rehovot, Israel.
6. Department of Cell and Molecular Physiology, Loyola University Chicago Stritch School of Medicine, Chicago, IL, USA.
7. Multiscale Instruments Inc., Seoul, Korea.

## Correspondence

<sup>\*</sup> Correspondence to:

- Dr. Shenhav Shemer: [shenhavc@technion.ac.il](mailto:shenhavc@technion.ac.il)
- Dr. Wonho Jhe: [whjhe@snu.ac.kr](mailto:whjhe@snu.ac.kr)

## Installation

The project requires Python 3.12 or later and uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage its environment and dependencies.

With Git and uv installed, clone the repository and create the environment:

```bash
git clone <repository-url> desmin-reconstruction
cd desmin-reconstruction
uv sync
```

Replace `<repository-url>` with this repository's GitHub URL. The `uv sync` command creates a local virtual environment in `.venv` and installs the project dependencies specified in `pyproject.toml`.

### GPU and CPU execution

The implementation uses JAX for GPU-accelerated computation. The default dependency specification, `jax[cuda12]>=0.8.0`, provides support for NVIDIA GPUs using CUDA 12. GPU execution requires compatible hardware and an NVIDIA driver supported by the installed JAX version; see the [JAX installation documentation](https://docs.jax.dev/en/latest/installation.html) for platform requirements.

The code also supports execution on a CPU using a CPU-only JAX installation, although computation times may be longer. For a CPU-only environment, replace `"jax[cuda12]>=0.8.0"` with `"jax>=0.8.0"` in `pyproject.toml` before running `uv sync`.

## Figure notebooks

Notebooks for generating the manuscript figures are located in [`notebooks/figures/`](notebooks/figures/). Open them in a Jupyter-compatible notebook interface and select the project's `.venv` environment as the Python kernel.

## License

This project's code is distributed under the MIT License. See [LICENSE](LICENSE) for details.
