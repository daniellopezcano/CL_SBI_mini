# CL_SBI_mini

Libraries to install in a new virtual enviroment:

conda create -n VE_CL_SBI_mini python=3.9
conda activate VE_CL_SBI_mini

conda install numpy scipy h5py matplotlib iminuit jupyter
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install sbi
pip install baccoemu

git clone 
cd ./CL_SBI_mini/
pip install -e .

screen -S JN
conda activate VE_CL_SBI_mini
jupyter notebook



