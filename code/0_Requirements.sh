conda create -n cvae python=3.9 pytorch torchvision torchaudio pandas scikit-learn pytorch-cuda=11.7 -c pytorch -c nvidia -c conda-forge
conda activate cvae
pip install biobricks
pip install dvc
pip install tqdm
pip install rdkit
pip install matplotlib
pip install dask
python -m pip install "dask[distributed]"