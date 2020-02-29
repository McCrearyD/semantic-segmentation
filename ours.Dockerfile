# NVIDIA Pytorch Image
FROM nvcr.io/nvidia/pytorch:19.05-py3

RUN pip install numpy
RUN pip install sklearn
RUN pip install h5py
RUN pip install jupyter
RUN pip install scikit-image
RUN pip install pillow
RUN pip install piexif
RUN pip install cffi
RUN pip install tqdm
RUN pip install dominate
RUN pip install tensorboardX
RUN pip install opencv-python
RUN pip install nose
RUN pip install ninja

RUN apt-get update
RUN apt-get install libgtk2.0-dev -y && rm -rf /var/lib/apt/lists/*


# Install Apex
RUN cd /home/ && git clone https://github.com/NVIDIA/apex.git apex && cd apex && python setup.py install --cuda_ext --cpp_ext

ADD requirements.txt /home/

RUN git clone https://github.com/McCrearyD/semantic-segmentation.git
RUN mkdir outputs
RUN mkdir models

# our weights
WORKDIR /home/models
RUN wget "http://kurtulm.us/models/kitti_best.pth"
RUN wget "http://kurtulm.us/models/cityscapes_best.pth"

WORKDIR /home/

RUN pip install -r requirements.txt