# docker build -t insilica/chemsim .
FROM nvidia/cuda:11.7.1-devel-ubuntu20.04

# Set a noninteractive frontend to prevent prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, including Python 3.9, pip, and Git
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-dev \
    git

# Symlink Python 3.9 as the default Python version
RUN ln -s /usr/bin/python3.9 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch and other dependencies
RUN pip install \
    torch torchvision torchaudio \
    pandas scikit-learn \
    biobricks dvc tqdm \
    matplotlib dask[distributed] \
    rdkit

# RDKIT needs libxrender1
RUN apt-get install -y libxrender1

# Install the requirements
COPY flask-cvae/requirements.txt requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Expose the port the app runs on
EXPOSE 6515

# Command to run the application
ENV FLASK_APP=flask-cvae.app
ENV ROOT_URL=http://localhost:6515

# Start the container with a bash shell
CMD ["gunicorn", "-b", "0.0.0.0:6515", "flask-cvae.app:app"]