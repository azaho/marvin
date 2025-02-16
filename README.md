Optionally, create a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

To start working with the project, install dependencies:

In case you are working with Nvidia RTX5080, install PyTorch form the latest nightly build:

```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

otherwise, just install normal PyTorch

```
pip install torch torchvision
```

And then install the rest of the libraries:

```
pip install dandi pynwb h5py matplotlib scikit-learn pygame dynamixel_sdk ikpy psutil ipykernel
```

Then, download the dataset from Churchland et al. (https://dandiarchive.org/dandiset/000070):

```
python jenkins_data_download.py
```
