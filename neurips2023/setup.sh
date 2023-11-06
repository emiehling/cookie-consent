#!/bin/bash

# Install the forked version of scikit-learn
git clone https://github.com/scikit-learn/scikit-learn.git
cd scikit-learn
git fetch origin pull/8474/head:nmf_missing
git checkout nmf_missing
pip install .
cd ..

# Install the packages defined by requirements.txt in the root directory
pip install -r requirements.txt

# Clone and install recsim from GitHub
git clone https://github.com/google-research/recsim.git
cd recsim
pip install -e .
#pip uninstall dopamine-rl -y
cd ..