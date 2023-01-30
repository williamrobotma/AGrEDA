#!/usr/bin/env bash

install_dir="${1:-./}"
venv_folder_name="${2:-.venv}"

install_dir=$(realpath "${install_dir}")
venv_dir=$(realpath "${install_dir}/${venv_folder_name}")
start_dir=$PWD

module load python/3.8
if [ -d "${venv_dir}" ]; then
    rm -r "${venv_dir}"
fi
virtualenv --no-download "${venv_dir}"
source "${venv_dir}/bin/activate"
pip install --no-index --upgrade pip


mkdir -p "${install_dir}/wheels"
if ! [ -f "${install_dir}/wheels/matplotlib-venn-0.11.7.tar.gz" ]; then
    pip download --no-deps matplotlib-venn==0.11.7 -d "${install_dir}/wheels"
fi
if ! [ -f "${install_dir}/wheels/harmonypy-0.0.9-py3-none-any.whl" ]; then
    pip download --no-deps harmonypy==0.0.9 -d "${install_dir}/wheels"
fi
if ! [ -f "${install_dir}/wheels/imbalanced_learn-0.10.1-py3-none-any.whl" ]; then
    pip download --no-deps imbalanced-learn==0.10.1 -d "${install_dir}/wheels"
fi

cp requirements_cc.txt "${install_dir}/tmp_agreda_requirements_cc.txt"
sed -i "s;\./;${install_dir}/;" "${install_dir}/tmp_agreda_requirements_cc.txt"
pip install --no-index -r "${install_dir}/tmp_agreda_requirements_cc.txt"
rm "${install_dir}/tmp_agreda_requirements_cc.txt"

echo -e '#!/bin/bash\nunset XDG_RUNTIME_DIR\njupyter lab --ip $(hostname -f) --no-browser' > $VIRTUAL_ENV/bin/jupyterlab.sh
chmod u+x $VIRTUAL_ENV/bin/jupyterlab.sh


