# Install MFA
INSTALL_DIR = "/tmp/mfa" # pass to install directory
bash ./install_mfa.sh {INSTALL_DIR}
source {INSTALL_DIR}/miniconda3/bin/activate aligner; mfa align --help

root_dir=${1:-/tmp/mfa}
mkdir -p $root_dir
cd $root_dir

# download miniconda3
wget -q --show-progress https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $root_dir/miniconda3 -f

# create py38 env
$root_dir/miniconda3/bin/conda create -n aligner -c conda-forge openblas python=3.8 openfst pynini ngram baumwelch -y
source $root_dir/miniconda3/bin/activate aligner

# install mfa, download kaldi
pip install montreal-forced-aligner # install requirements
pip install git+https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner.git # install latest updates

mfa thirdparty download

echo -e "\n======== DONE =========="
echo -e "\nTo activate MFA, run: source $root_dir/miniconda3/bin/activate aligner"
echo -e "\nTo delete MFA, run: rm -rf $root_dir"
echo -e "\nSee: https://montreal-forced-aligner.readthedocs.io/en/latest/aligning.html to know how to use MFA"

# download and install mfa
INSTALL_DIR="/tmp/mfa" # path to install directory

!bash ./install_mfa.sh {INSTALL_DIR}
!source {INSTALL_DIR}/miniconda3/bin/activate aligner; mfa align --help