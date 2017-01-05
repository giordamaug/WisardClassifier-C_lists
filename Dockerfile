############################################################
# Dockerfile to build BEWiS container images
# Based on Ubuntu
############################################################

# Set the base image to Ubuntu
FROM ubuntu:14.04

# File Author / Maintainer
MAINTAINER Maurizio Giordano

# Update the repository sources list
RUN apt-get update

################## PYTHON INSTALLATION ######################
# Install Python and require graphic libraries for Opencv
#

ENV PYTHON_VERSION 3.5
ENV NUM_CORES 4

# Install OpenCV 3.0
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    build-essential \
    ca-certificates \
    gcc \
    cmake \
    git \
    libpq-dev \
    make \
    mercurial \
    pkg-config \
    python \
    python-dev \
    python-pip
RUN pip install --upgrade pip
RUN pip install numpy scipy scikit-learn pillow

################## WISC INSTALLATION ######################
# Install WisardClassifier
#

WORKDIR /home
RUN git clone https://github.com/giordamaug/WisardLibrary
WORKDIR /home/WisardLibrary
RUN cmake .
RUN make
WORKDIR /home
RUN git clone https://github.com/giordamaug/WisardClassifier
WORKDIR /home/WisardClassifier
RUN python test.py
