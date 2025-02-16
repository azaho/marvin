Optionally, create a virtual environment:

```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

To start working with the project, install dependencies:

```
pip install dandi pynwb h5py matplotlib torch torchvision scikit-learn pygame dynamixel_sdk ikpy psutil
```

Then, download the dataset from Churchland et al. (https://dandiarchive.org/dandiset/000070):

```
python jenkins_data_download.py
```
