pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install --editable ./
python setup.py build develop
pip install seaborn omegaconf matplotlib tensorboardX scipy

git config --global user.name v-lbei
git config --global user.email FAREAST.v-lbei@microsoft.com


sudo mkdir -p /scratch/blobfuse/f_xdata
sudo mkdir -p /scratch/blobfuse/f_xdata/G/mmt/
sudo mkdir -p /scratch/blobfuse/f_xdata/G/mmt/checkpoints
sudo chmod -R 777 /scratch/blobfuse/f_xdata


cd ~/fairseq-msra
ln -s /scratch/blobfuse/f_xdata/G/mmt/checkpoints ~/fairseq-msra/checkpoints

azcopy copy "https://chenfei.blob.core.windows.net/data/G/mmt/data-bin/?sv=2020-10-02&st=2022-09-25T04%3A34%3A31Z&se=2024-09-26T04%3A34%3A00Z&sr=c&sp=racwdxl&sig=qhi5AaczGeVZnUnaBe%2F0jpes6LYqMlAqoqOHhH7TkuE%3D" "./"   --recursive
