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

RUN pip3 install transitions

# Fix für mDNS: Erlaubt dem Container, .local Namen aufzulösen
RUN sed -i 's/hosts: files dns/hosts: files mdns4_minimal [NOTFOUND=return] dns mdns4/' /etc/nsswitch.conf

# Automatisches ROS Sourcing für jedes neue Terminal
RUN echo 'export ROS_IP=$(hostname -I | awk "{print \$1}")' >> /root/.bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN echo "[ -f /root/DuckieRace/devel/setup.bash ] && source /root/DuckieRace/devel/setup.bash" >> /root/.bashrc
RUN echo "alias dt='bash /root/DuckieRace/launch.sh'" >> /root/.bashrc
