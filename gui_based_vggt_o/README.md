<div align="center">
<h1>VGGT-&Omega;</h1>

<a href="http://vggt-omega.github.io/" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2605.15195" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/arXiv-2605.15195-b31b1b" alt="arXiv"></a>
<a href="https://huggingface.co/spaces/facebook/vggt-omega"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>

<p>
  <span class="author"><a href="https://jytime.github.io/">Jianyuan Wang</a><sup>1,2</sup></span>
  <span class="author"><a href="https://silent-chen.github.io/">Minghao Chen</a><sup>1</sup></span>
  <span class="author"><a href="https://scholar.google.com/citations?user=FUDsZkEAAAAJ&amp;hl=zh-CN">Shangzhan Zhang</a><sup>1</sup></span>
  <span class="author"><a href="https://nikitakaraevv.github.io/">Nikita Karaev</a><sup>1</sup></span>
  <br>
  <span class="author"><a href="https://demuc.de/">Johannes Schönberger</a><sup>2</sup></span>
  <span class="author"><a href="https://scholar.google.com/citations?user=IJidh-UAAAAJ&amp;hl=fr">Patrick Labatut</a><sup>2</sup></span>
  <span class="author"><a href="https://scholar.google.com/citations?user=lJ_oh2EAAAAJ&amp;hl=en">Piotr Bojanowski</a><sup>2</sup></span>
  <span class="author"><a href="https://d-novotny.github.io/">David Novotny</a></span>
  <br>
  <span class="author"><a href="https://www.robots.ox.ac.uk/~vedaldi/">Andrea Vedaldi</a><sup>1,2</sup></span>
  <span class="author"><a href="https://chrirupp.github.io/">Christian Rupprecht</a><sup>1</sup></span>
</p>

**<sup>1</sup>[Visual Geometry Group, University of Oxford](https://www.robots.ox.ac.uk/~vgg/)**; **<sup>2</sup>[Meta AI](https://ai.facebook.com/research/)**
</div>

## Pretrained models

Before using the models, please request access to the checkpoints [here](https://huggingface.co/facebook/VGGT-Omega). Once your request is approved, you can download the checkpoints. Please note that access requests are reviewed by an automated process based on the information provided in the request.

| Model | Resolution | Text alignment | Download |
| :--- | :--- | :--- | :--- |
| `VGGT-Omega-1B-512` | 512 | No | [Link](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_512.pt) |
| `VGGT-Omega-1B-256-Text-Alignment` | 256 | Yes | [Link](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_256_text.pt) |

The authors are not involved in the review process and cannot approve or reject individual applications. However, the [🤗 Hugging Face demo](https://huggingface.co/spaces/facebook/vggt-omega) is available to everyone.


## Quick Start

First, clone this repository and install the dependencies:

```bash
git clone git@github.com:facebookresearch/vggt-omega.git
cd vggt-omega
pip install -r requirements.txt
pip install -e .
```


Now, try the model with a few lines of code:

```python
import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera

checkpoint_path = "path/to/vggt_omega_1b_512.pt"
image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]

model = VGGTOmega().to("cuda").eval()
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

images = load_and_preprocess_images(image_names, image_resolution=512).to("cuda")

with torch.inference_mode():
    predictions = model(images)

extrinsics, intrinsics = encoding_to_camera(
    predictions["pose_enc"],
    predictions["images"].shape[-2:],
)

depth = predictions["depth"]
depth_conf = predictions["depth_conf"]
camera_and_register_tokens = predictions["camera_and_register_tokens"]
camera_tokens = camera_and_register_tokens[:, :, :1]
registers = camera_and_register_tokens[:, :, 1:]
```

For the text-aligned checkpoint, use `VGGTOmega(enable_alignment=True)` with `image_resolution=256` and read `predictions["text_alignment_embedding"]`.

## Export Anipose calibration

After your Hugging Face account has access to the VGGT-Omega checkpoint, install
the current Hugging Face CLI and download the weights:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login
mkdir -p checkpoints
hf download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir checkpoints
```

Then convert a set of camera predictions into an Anipose/aniposelib-compatible
`calibration.toml`:

```bash
python scripts/export_anipose_calibration.py \
  --checkpoint checkpoints/vggt_omega_1b_512.pt \
  --output calibration.toml \
  --names A B C \
  --image-size 1200x800 \
  pA.png pB.png pC.png
```

If `--image-size` is omitted, the script uses each input image's actual
`WIDTHxHEIGHT`. Camera names are inferred from filenames such as `pA.png` or
`camA.png`, but `--names` is safer when the downstream Anipose camera IDs must
match existing project folders. Use `--image-dir` or `--image-glob` instead of
positional paths when the image set should be discovered from disk.

VGGT-Omega predicts pinhole intrinsics and camera-from-world poses; it does not
estimate lens distortion, so the exported distortion coefficients default to
zeros. Use already-undistorted images when possible, or pass known coefficients
with `--distortions`.

For the synchronized video calibration workflow used in this workspace, see
[`vggt_calibration_pipeline/README.md`](./vggt_calibration_pipeline/README.md).
That pipeline extracts frames with OpenCV, runs source-sized VGGT-Omega
calibration, compares calibration TOMLs, and optionally evaluates the selected
calibrations through a separate Anipose environment.

## Interactive Demo

Install the demo dependencies:

```bash
pip install -r requirements_demo.txt
```

Launch the Gradio demo with a local checkpoint path:

```bash
python demo_gradio.py \
  --checkpoint checkpoints/VGGT-Omega-1B-512/model.pt \
  --image-resolution 512
```

The demo accepts uploaded images or a video, runs camera and depth inference,
and visualizes the depth-unprojected point cloud and predicted cameras as a GLB
scene.

## Runtime and GPU Memory

We benchmark the end-to-end peak GPU memory usage of `VGGT-Omega-1B-512` on a
single NVIDIA A100 GPU with 624x416 input images. The measurement covers the full
inference program, from loading the model weights onto the GPU through the
forward pass, so it includes both the memory needed to store the model itself
and the memory used by inference activations and buffers. In other words, a GPU
with at least the listed available memory is able to run the corresponding
number of input frames under this setup.

| **Input Frames** | 1 | 10 | 25 | 50 | 100 | 200 | 300 | 400 | 500 |
|:----------------:|:-:|:--:|:--:|:--:|:---:|:---:|:---:|:---:|:---:|
| **Peak Memory (GB)** | 6.02 | 6.67 | 7.80 | 9.66 | 13.37 | 20.82 | 28.26 | 35.71 | 43.15 |

The benchmark uses [`load_and_preprocess_images`](./vggt_omega/utils/load_fn.py)
with the default `mode="balanced"` and `image_resolution=512`. For these roughly
3:2 landscape images, this produces 624x416 inputs. You can set
`mode="max_size"` to resize the longest side to 512 instead; for the same aspect
ratio, this gives about 512x336 inputs and uses less GPU memory.

## License

See the [LICENSE](./LICENSE) file for details about the license under which
this code is made available.

[^release]: This Release is intended to support the open source research community.

```bibtex
@inproceedings{wang2026vggtomega,
  title={VGGT-{$\Omega$}},
  author={Wang, Jianyuan and Chen, Minghao and Zhang, Shangzhan and Karaev, Nikita and Sch{\"o}nberger, Johannes and Labatut, Patrick and Bojanowski, Piotr and Novotny, David and Vedaldi, Andrea and Rupprecht, Christian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```
