#!/bin/bash

cuda_version="$1"
os="$2"

# Need ubuntu 20.04 for this which is not supported anymore
# "cu113")
#   url=https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.1-465.19.01-1_amd64.deb
#   ;;

case "${cuda_version}" in
  "cu117")
    url=https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda-repo-ubuntu2204-11-7-local_11.7.1-515.65.01-1_amd64.deb
    ;;
  "cu118")
    url=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
    ;;
  "cu121")
    url=https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.1-530.30.02-1_amd64.deb
    ;;
  "cu124")
    url=https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda-repo-ubuntu2204-12-4-local_12.4.1-550.54.15-1_amd64.deb
    ;;
  "cu126")
    url=https://developer.download.nvidia.com/compute/cuda/12.6.3/local_installers/cuda-repo-ubuntu2204-12-6-local_12.6.3-560.35.05-1_amd64.deb
    ;;
  *)
    >&2 echo "Unsupported cuda_version: ${cuda_version}"
    exit 1
    ;;
esac

if [[ "${os}" != "Linux" ]]; then
  >&2 echo "Unsupported OS: ${os}"
  exit 1
fi

wget "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin"
sudo mv "cuda-ubuntu2204.pin" /etc/apt/preferences.d/cuda-repository-pin-600
wget "${url}"
sudo dpkg -i "$(basename "${url}")"

keyfile=$(ls /var/cuda-repo-ubuntu2204-*/cuda-*-keyring.gpg)
if [[ -f "${keyfile}" ]]; then
    sudo cp "${keyfile}" /usr/share/keyrings/
  else
    sudo apt-key add /var/cuda-repo-*/7fa2af80.pub
fi

sudo apt-get update
sudo apt-get -y install cuda

/usr/local/cuda/bin/nvcc --version
