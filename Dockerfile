# NVIDIA Pytorch Image
FROM nvcr.io/nvidia/pytorch:19.05-py3

# our weights
RUN mkdir models
WORKDIR /home/models
RUN wget "http://kurtulm.us/models/kitti_best.pth" \
    && wget "http://kurtulm.us/models/cityscapes_best.pth" \
    && wget "http://kurtulm.us/models/wider_resnet38.pth.tar"
WORKDIR /home/

RUN apt-get update
RUN apt-get install libgtk2.0-dev -y && rm -rf /var/lib/apt/lists/*

# Install Apex
RUN cd /home/ && git clone https://github.com/NVIDIA/apex.git apex && cd apex && python setup.py install --cuda_ext --cpp_ext

WORKDIR /home/
ADD requirements.txt /home/
RUN pip install -r requirements.txt

RUN mkdir semantic-segmentation
ADD ./ /home/semantic-segmentation

RUN echo 'echo "#### (1) Command for running the jupyter notebook: jupyter-notebook --port 8080 --ip 0.0.0.0 --allow-root"' >> /root/.bashrc