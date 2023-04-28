conda create -n ptvae python=3.10 -y
conda activate ptvae
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install pandas numpy matplotlib tqdm -y
conda install -c conda-forge scikit-learn -y
pip install h5py argparse
