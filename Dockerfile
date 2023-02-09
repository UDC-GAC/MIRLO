FROM ubuntu:20.04

# Install R
RUN apt-get update
RUN apt-get install -y dirmngr gnupg apt-transport-https ca-certificates \
                       software-properties-common libcurl4-openssl-dev \
                       libssl-dev libxml2-dev
RUN apt-key adv --keyserver keyserver.ubuntu.com \
                --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'
RUN apt-get update
RUN apt-get install -y r-base r-base-dev r-base-core

# Install Python
RUN apt-get install -y python3 python3-pip python3-venv

# Install sudo
RUN apt-get install -y sudo

# Create mirlo user and copy files
RUN useradd -ms /bin/bash mirlo
RUN usermod -aG sudo mirlo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

WORKDIR /home/mirlo/MIRLO
COPY . .

RUN chmod 777 /usr/local/lib/R/site-library
RUN chown -R mirlo:mirlo /home/mirlo
USER mirlo

# Install LION and its dependencies
RUN MAKE="make -j$(getconf _NPROCESSORS_ONLN)" \
    Rscript -e "install.packages(c('seqinr', 'randomForest', 'caret', 'RCurl'))"
RUN MAKE="make -j$(getconf _NPROCESSORS_ONLN)" \
    Rscript -e "install.packages('misc/LION_0.2.8.tar.gz', repos = NULL, type = 'source')"

# Intall MIRLO dependencies
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

CMD ["bash"]
