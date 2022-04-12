# Use an official Python runtime as a parent image
FROM tensorflow/tensorflow:2.5.0-gpu
RUN echo

# User configuration - override with --build-arg
ARG user=myuser
ARG group=mygroup
ARG uid=1000
ARG gid=1000

# Some debs want to interact, even with apt-get install -y, this fixes it
ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/project

# Install any needed packages from apt
RUN apt-get update && apt-get install -y sudo python3 python3-pip git screen vim
RUN pip install --upgrade pip

# Configure user
RUN groupadd -g $gid $user
RUN useradd -u $uid -g $gid $user
RUN usermod -a -G sudo $user
RUN passwd -d $user

RUN apt-get install -y libsm6 libxrender1 ffmpeg libsm6 libxext6

# Need to do both installs to make it work
RUN git clone https://github.com/vaneeda/deepvision-retinanet.git
RUN cd deepvision-retinanet && python setup.py build_ext --inplace
RUN cd deepvision-retinanet && pip3 install . 

RUN pip3 install --trusted-host pypi.python.org numpy scipy Pillow cython matplotlib scikit-image keras==2.4.0 opencv-python h5py imgaug IPython progressbar2 pandas sklearn

COPY /snapshots/resnet50_csv_inference_10.h5 /snapshots/resnet50_csv_inference_10.h5

RUN pip3 install lxml
COPY csv2xml_docker.py /csv2xml_docker.py

# Run when the container launches
# CMD "bash"
#
WORKDIR /deepvision-retinanet
CMD ["/bin/bash", "-c", "python3 keras_retinanet/bin/predict_DV.py && python3 /csv2xml_docker.py"]
#RUN cd /deepvision-retinanet && python3 keras_retinanet/bin/predict_DV.py