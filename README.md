## Software environment:
```
# Python with the following version and libraries.

conda create -n mae-net python=3.9

conda activate mae-net 

conda install tensorflow-gpu=2.4.1

conda install scikit-image tqdm scikit-learn pydot

conda install -c conda-forge oyaml

pip install tensorflow-addons==0.13.0

pip install numpy==1.19.2
```

## Citation

If you use the provided method in this repository, please cite the following paper:

```
   title={Multitask Adversarial Networks Based on Extended Nonlinear Spiking Neuron Models}, 
   author={Jun Fu, Hong Peng, Bing Li, Zhicai Liu, Rikong Lugu, Jun Wang, and Antonio Ramírez-de-Arellano},
   year={2024},
```
The article has been accepted.

## Downloading the Dataset

We collect the dataset consisting of unpaired poor and good quality X-ray samples.
The dataset is described in our article.
please download the file and unzip it:
```
unzip dataset.zip
```

## Training

The proposed MAE-Net method can be trained as follows,
```
python train.py --method operational --q 3
```

## Testing

```
python test.py --method operational --q 3
```
Optionally you can use ```--saveImages``` flag, e.g., ```python test.py --method operational --q 3 --saveImages True``` to store the restored images by the selected model under ```output/samples_testing``` folder.
