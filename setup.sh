#!/usr/bin/env bash
# Set up and build llm_sct on a fresh Ubuntu 22.04 PC.

set -Eeuo pipefail

readonly ROS_DISTRO="humble"
readonly REQUIRED_CODENAME="jammy"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${EUID}" -eq 0 ]]; then
  echo "Run this script as your normal user, not with sudo." >&2
  exit 1
fi

if [[ ! -r /etc/os-release ]]; then
  echo "Cannot identify the operating system. Ubuntu 22.04 is required." >&2
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_CODENAME:-}" != "${REQUIRED_CODENAME}" ]]; then
  echo "This setup targets Ubuntu 22.04 (${REQUIRED_CODENAME})." >&2
  echo "Detected: ${PRETTY_NAME:-unknown operating system}." >&2
  exit 1
fi

# Support both recommended layouts:
#   ~/sct_ws/src/llm_sct/setup.sh
#   ~/sct_ws/setup.sh  (repository cloned as the workspace)
if [[ "$(basename -- "$(dirname -- "${SCRIPT_DIR}")")" == "src" ]]; then
  WORKSPACE_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
else
  WORKSPACE_DIR="${SCRIPT_DIR}"
fi

echo "Installing Ubuntu and ROS 2 tools..."
sudo apt-get update
sudo apt-get install -y curl git locales software-properties-common
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
sudo add-apt-repository -y universe

if [[ ! -f /etc/apt/sources.list.d/ros2.list ]]; then
  echo "Adding the official ROS 2 apt repository..."
  sudo curl -fsSL \
    https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu ${REQUIRED_CODENAME} main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
fi

sudo apt-get update
sudo apt-get install -y \
  ros-humble-desktop \
  ros-dev-tools \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool

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
  --from-paths "${WORKSPACE_DIR}" \
  --ignore-src \
  --rosdistro "${ROS_DISTRO}" \
  --skip-keys "leo_description leo_gz_bringup leo_gz_plugins leo_gz_worlds leo_image_processing leo_supervisor_common" \
  --reinstall \
  -r -y

echo "Building the workspace at ${WORKSPACE_DIR}..."
cd "${WORKSPACE_DIR}"
colcon build --symlink-install

echo
echo "Setup complete. In each new terminal, run:"
echo "  source /opt/ros/${ROS_DISTRO}/setup.bash"
echo "  source ${WORKSPACE_DIR}/install/setup.bash"
