## Generating Prompts in Latet Space for Rehearsal-free Continual Learning
### Requirements

Our experiments are done with:

- python 3.11.8
- pytorch 2.2.1
- tensorflow 2.16.1
- numpy 1.26.4
- fvcore 0.1.5.post20221221
- tensorflow-datasets  4.9.4
- scipy 1.12.0
- ml-collections 0.1.1

### Environment setup
```bash
conda create -n gpls python=3.11
conda activate gpls
bash env_setup.sh
```

### Pretraiend ViT model

- ViT-B/16 model used in GPLS can be downloaded at https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz.
- Remember to rename it from `ViT-B_16.npz`  to `imagenet21k_ViT-B_16.npz`.

~~~bash
cd model
mv ViT-B_16.npz imagenet21k_ViT-B_16.npz
~~~

- The pre-trained ViT-B/16 checkpoint is expected to be situated in the 'model' folder.

### Data preparation

- CIFAR-100, DomainNet, EuroSAT are downloaded automatically, while Oxford-IIIT Pets and CropDisease can only be downloaded manually because TensorFlow have not provide the corresponding library. 
- Oxford-IIIT Pets and CropDisease can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1bBqS8MuTQXUBV3DXJ_-YZyNOR4ejvm1O?usp=sharing).
- Please remember to put the downloaded data into the 'dataset' folder. 
- We transform [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/) and [CropDiseases](https://www.frontiersin.org/articles/10.3389/fpls.2016.01419/full) into TFDS compatible form following the tutorial in [link](https://www.tensorflow.org/datasets/add_dataset) to cover the CL scenario (see sec. 5 and supp. A for details). 

### Run experiments
The execution code on different datasets is as follows
```
python train.py --config-file configs/dap/cifar.yaml
python train.py --config-file configs/dap/domainnet.yaml
python train.py --config-file configs/dap/pets.yaml
python train.py --config-file configs/dap/curosat.yaml
python train.py --config-file configs/dap/cropdisease.yaml
```

