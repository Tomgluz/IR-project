<<<<<<< HEAD
#!/bin/bash
set -e

APP_USER="tomgluz"
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  useradd -m -s /bin/bash "$APP_USER"
fi

APP_HOME="/home/${APP_USER}"
VENV_DIR="${APP_HOME}/venv"

apt-get update
apt-get install -y python3-venv python3-pip

# Create venv as the normal user (not root)
sudo -u "${APP_USER}" bash -lc "
python3 -m venv '${VENV_DIR}'
source '${VENV_DIR}/bin/activate'
pip install --upgrade pip
pip install \
  'Flask==2.0.2' \
  'Werkzeug==2.3.8' \
  'flask-restful==0.3.9' \
  'nltk==3.6.3' \
  'pandas' \
  'google-cloud-storage' \
  'numpy>=1.23.2,<3'
=======
#!/bin/bash
set -e

APP_USER="tomgluz"
if ! id -u "$APP_USER" >/dev/null 2>&1; then
  useradd -m -s /bin/bash "$APP_USER"
fi

APP_HOME="/home/${APP_USER}"
VENV_DIR="${APP_HOME}/venv"

apt-get update
apt-get install -y python3-venv python3-pip

# Create venv as the normal user (not root)
sudo -u "${APP_USER}" bash -lc "
python3 -m venv '${VENV_DIR}'
source '${VENV_DIR}/bin/activate'
pip install --upgrade pip
pip install \
  'Flask==2.0.2' \
  'Werkzeug==2.3.8' \
  'flask-restful==0.3.9' \
  'nltk==3.6.3' \
  'pandas' \
  'google-cloud-storage' \
  'numpy>=1.23.2,<3'
>>>>>>> 688a912263a363b7a7d41e78519ee995403f7c85
"