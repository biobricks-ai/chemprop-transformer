# docker build -t biobricks-ai/cvae .
# docker run -p 6515:6515 -v .:/chemsim --rm --gpus all -it --name chemsim biobricks-ai/cvae
# curl "http://localhost:6515/predict?inchi=InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)" 
FROM nvidia/cuda:12.3.1-base-ubuntu20.04

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
COPY flask_cvae/requirements.txt requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the necessary resources
# COPY brick/moe brick/moe
# COPY brick/cvae.sqlite brick/cvae.sqlite
# COPY brick/selfies_property_val_tokenizer brick/selfies_property_val_tokenizer

# Expose the port the app runs on
EXPOSE 6515

# Command to run the application
ENV FLASK_APP=flask_cvae.app
ENV ROOT_URL=http://localhost:6515

# Start the container with a bash shell
WORKDIR /chemsim
CMD ["gunicorn", "-b", "0.0.0.0:6515", "--timeout", "480", "--graceful-timeout", "480", "--workers", "1", "flask_cvae.app:app"]