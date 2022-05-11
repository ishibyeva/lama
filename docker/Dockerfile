FROM ubuntu:20.04

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget mc tmux nano build-essential rsync libgl1

ARG USERNAME=user
RUN apt-get install -y sudo && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME

USER $USERNAME:$USERNAME
WORKDIR "/home/$USERNAME"
ENV PYTHONPATH="/home/$USERNAME/project"

RUN pip install -U pip
RUN pip install numpy scipy torch==1.8.1 torchvision opencv-python tensorflow joblib matplotlib pandas \
    albumentations==0.5.2 pytorch-lightning==1.2.9 tabulate easydict==1.9.0 kornia==0.5.0 webdataset \
    packaging gpustat tqdm pyyaml hydra-core==1.1.0.dev6 scikit-learn==0.24.2 tabulate pytest==7.1.2 
    
RUN pip install scikit-image==0.17.2

ENV TORCH_HOME="/home/$USERNAME/.torch"

CMD [ "/bin/bash", "4_testing.sh" ]
