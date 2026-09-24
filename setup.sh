#!/usr/bin/env bash
# Set up and build llm_sct with Gazebo GUI support on Ubuntu 24.04.

set -Eeuo pipefail

readonly ROS_DISTRO="jazzy"
readonly REQUIRED_CODENAME="noble"
readonly GAZEBO_RELEASE="harmonic"
readonly GAZEBO_SIM_MAJOR="8"
readonly OPENAI_PYTHON_VERSION="1.90.0"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this script as your normal user, not with sudo." >&2
  exit 1
fi

if [[ ! -r /etc/os-release ]]; then
  echo "Cannot identify the operating system. Ubuntu 24.04 is required." >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_CODENAME:-}" != "${REQUIRED_CODENAME}" ]]; then
  echo "This setup targets Ubuntu 24.04 (${REQUIRED_CODENAME})." >&2
  echo "Detected: ${PRETTY_NAME:-unknown operating system}." >&2
  exit 1
fi

# Gazebo Harmonic is the supported Gazebo release paired with ROS 2 Jazzy.
# This is also consumed by the conditional dependencies and CMake logic in
# leo_gz_plugins.
export GZ_VERSION="${GAZEBO_RELEASE}"

# Support both recommended layouts:
#   ~/sct_ws/src/llm_sct/setup.sh
#   ~/sct_ws/setup.sh  (repository cloned as the workspace)
if [[ "$(basename -- "$(dirname -- "${SCRIPT_DIR}")")" == "src" ]]; then
  WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
  PACKAGE_PATH="${WORKSPACE_DIR}/src"
else
  WORKSPACE_DIR="${SCRIPT_DIR}"
  PACKAGE_PATH="${SCRIPT_DIR}"
fi

echo "Installing Ubuntu and ROS 2 tools..."
sudo apt-get update
sudo apt-get install -y curl git locales software-properties-common
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
sudo add-apt-repository -y universe

echo "Configuring the official ROS 2 apt repository for ${REQUIRED_CODENAME}..."
sudo curl -fsSL \
  https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${REQUIRED_CODENAME} main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null

sudo apt-get update
sudo apt-get install -y \
  "ros-${ROS_DISTRO}-desktop" \
  "ros-${ROS_DISTRO}-ros-gz" \
  "ros-${ROS_DISTRO}-cv-bridge" \
  "ros-${ROS_DISTRO}-image-transport" \
  ros-dev-tools \
  python3-colcon-common-extensions \
  python3-matplotlib \
  python3-numpy \
  python3-opencv \
  python3-pip \
  python3-rosdep \
  python3-tk \
  python3-venv \
  python3-vcstool \
  python3-yaml \
  libopencv-dev \
  ffmpeg

# Ubuntu 24.04 treats its system Python as externally managed. Keep the OpenAI
# dependency in a project virtual environment while retaining access to the
# apt-installed ROS Python packages.
VENV_DIR="${WORKSPACE_DIR}/.venv-${ROS_DISTRO}"
python3 -m venv --system-site-packages "${VENV_DIR}"
touch "${VENV_DIR}/COLCON_IGNORE"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
python3 -m pip install --upgrade pip
python3 -m pip install "openai==${OPENAI_PYTHON_VERSION}"

# ROS 2 Jazzy targets Gazebo Harmonic, whose gz-sim major version is 8.
INSTALLED_GAZEBO_VERSION="$(gz sim --versions | head -n 1)"
INSTALLED_GAZEBO_MAJOR="$(grep -Eo '[0-9]+(\.[0-9]+){1,2}' <<<"${INSTALLED_GAZEBO_VERSION}" | head -n 1 | cut -d. -f1)"
if [[ "${INSTALLED_GAZEBO_MAJOR}" != "${GAZEBO_SIM_MAJOR}" ]]; then
  echo "Expected Gazebo Sim ${GAZEBO_SIM_MAJOR}.x (${GAZEBO_RELEASE}), but found ${INSTALLED_GAZEBO_VERSION}." >&2
  exit 1
fi
echo "Using Gazebo Sim ${INSTALLED_GAZEBO_VERSION} (${GAZEBO_RELEASE})."

python3 - <<'PY'
import cv2
import matplotlib
import numpy
import openai
import tkinter
import yaml

print(f"OpenCV {cv2.__version__}")
print(f"NumPy {numpy.__version__}")
print(f"Matplotlib {matplotlib.__version__}")
print(f"OpenAI Python {openai.__version__}")
print(f"PyYAML {yaml.__version__}")
PY

# shellcheck disable=SC1091
source "/opt/ros/${ROS_DISTRO}/setup.bash"

if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
  sudo rosdep init
fi
rosdep update

if [[ -f "${SCRIPT_DIR}/dependencies.repos" ]]; then
  echo "Importing source dependencies..."
  mkdir -p "${WORKSPACE_DIR}/src"
  vcs import "${WORKSPACE_DIR}/src" < "${SCRIPT_DIR}/dependencies.repos"
fi

echo "Installing dependencies declared by the ROS packages..."
rosdep install \
  --from-paths "${PACKAGE_PATH}" \
  --ignore-src \
  --rosdistro "${ROS_DISTRO}" \
  --skip-keys "leo_description leo_gz_bringup leo_gz_plugins leo_gz_worlds leo_image_processing leo_supervisor_common" \
  -r -y

echo "Building the workspace at ${WORKSPACE_DIR}..."
cd "${WORKSPACE_DIR}"
# Keep Jazzy/Python 3.12 artifacts separate from any existing Humble/Python
# 3.10 build in the same workspace.
BUILD_DIR="build/${ROS_DISTRO}"
INSTALL_DIR="install/${ROS_DISTRO}"
LOG_DIR="log/${ROS_DISTRO}"
colcon --log-base "${LOG_DIR}" build \
  --build-base "${BUILD_DIR}" \
  --install-base "${INSTALL_DIR}" \
  --symlink-install

echo
echo "Setup complete. In each new terminal, run:"
echo "  source /opt/ros/${ROS_DISTRO}/setup.bash"
echo "  source ${VENV_DIR}/bin/activate"
echo "  export GZ_VERSION=${GAZEBO_RELEASE}"
echo "  source ${WORKSPACE_DIR}/${INSTALL_DIR}/setup.bash"
echo
echo "Example Gazebo GUI launch:"
echo "  ros2 launch leo_exploration leo_gz.launch.py headless:=false"
echo
echo "Example headless launch:"
echo "  ros2 launch leo_exploration leo_gz.launch.py headless:=true"
