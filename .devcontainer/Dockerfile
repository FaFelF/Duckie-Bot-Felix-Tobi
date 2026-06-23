FROM ros:noetic-ros-base

# Installation von Tools und mDNS Support
RUN apt-get update && apt-get install -y \
    python3-opencv \
    python3-pip \
    python3-tk \
    avahi-utils \
    libnss-mdns \
    iputils-ping \
    git \
    && rm -rf /var/lib/apt/lists/*

# opencv-python (NICHT -headless): bringt GUI-Backend mit -> cv2.imshow funktioniert.
# pupil-apriltags fuer detect_apriltag_node.
RUN pip3 install transitions onnxruntime-gpu opencv-python "numpy>=1.21.6" pupil-apriltags

# CUDA 11.8 Runtime + cuDNN 8.9 fuer onnxruntime-gpu (GPU-Inferenz).
# Treiber-Lib (libcuda.so) kommt zur Laufzeit ueber "--gpus all" vom Host.
# Lib-Pfade systemweit registrieren -> kein LD_LIBRARY_PATH noetig.
RUN pip3 install --no-cache-dir \
        nvidia-cuda-runtime-cu11 nvidia-cublas-cu11 nvidia-cudnn-cu11==8.9.2.26 \
        nvidia-cufft-cu11 nvidia-curand-cu11 nvidia-cusparse-cu11 \
        nvidia-cusolver-cu11 nvidia-cuda-nvrtc-cu11 \
 && python3 -c "import nvidia, os, glob; print('\n'.join(glob.glob(os.path.dirname(nvidia.__file__)+'/*/lib')))" \
        > /etc/ld.so.conf.d/nvidia-cuda.conf \
 && ldconfig

# Fix für mDNS: Erlaubt dem Container, .local Namen aufzulösen
RUN sed -i 's/hosts: files dns/hosts: files mdns4_minimal [NOTFOUND=return] dns mdns4/' /etc/nsswitch.conf

# Automatisches ROS Sourcing für jedes neue Terminal
RUN echo 'export ROS_IP=$(hostname -I | awk "{print \$1}")' >> /root/.bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "[ -f /root/DuckieRace/devel/setup.bash ] && source /root/DuckieRace/devel/setup.bash" >> /root/.bashrc
RUN echo "alias dt='bash /root/DuckieRace/launch.sh'" >> /root/.bashrc
