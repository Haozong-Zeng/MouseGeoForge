import importlib.util
import sys
from importlib.metadata import version,PackageNotFoundError
for name in ['torch','torchvision','numpy','pandas','scipy','scikit-learn','scikit-image','matplotlib','lightning','pytorch-lightning','lightning-utilities','torchmetrics','imaug','imagecorruptions-imaug','kornia','kornia-rs','jaxtyping','wadler-lindig','tensorboard','tensorboard-data-server','absl-py','grpcio','markdown','werkzeug','safetensors','transformers','huggingface-hub','tokenizers','regex','aiohttp','fsspec','shapely','h5py','moviepy']:
    try:print(name,version(name))
    except PackageNotFoundError:print(name,'MISSING')
if '--import' in sys.argv:
    import lightning_pose.data.datasets
    import lightning_pose.models
    import lightning_pose.losses
    import lightning_pose.train
    import torch
    print('LP_IMPORT_OK',torch.__version__,torch.cuda.is_available())
