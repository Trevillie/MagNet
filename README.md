# MagNet

Demo code for "MagNet: a Two-Pronged Defense against Adversarial Examples", by Dongyu Meng and Hao Chen, at CCS 2017.

The code demos black-box defense against Carlini's L2 attack of various confidences.
Other techniques proposed in the paper are also included in `defensive_models.py` and `worker.py`, but are not shown in the demo defense.
Attack implementations are not provided in this repository.

## Run the demo code:

0. Make sure you have Keras, Tensorflow, numpy, scipy, and matplotlib installed.
1. Clone the repository.
2. We provide demo attack data and classifier on [Dropbox](https://www.dropbox.com/s/2x509u80g5zkuea/MagNet_support_data.zip?dl=0) and [百度网盘](https://pan.baidu.com/s/1gfpcB5p) (密码: yzt4). Please download and put the unzipped files in `MagNet/`. You may also use your own data for test.
3. Train autoencoders with `python3 train_defense.py`.
4. Test the defense with `python3 test_defense.py .`
5. Defense performance is plotted in `graph/defense_performance.pdf`.
