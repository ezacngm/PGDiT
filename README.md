
## Installation
- Python3>=3.8, PyTorch>=1.8.0, torchvision>=0.7.0 are required for the current codebase.
- To install the other dependencies, run
```bash
- conda env create -f environment.yaml 
- conda activate PGDiT
```

## Prepare data and pretrained model
**Dataset:**  
We use the same data & processing strategy following U-Mamba. Download dataset from HCP offical website.
We provide data processing ipynb, run gen_dataset.ipynb

**HCP-pretrained model:**  
We provide the model checkpoint of PGDiT size XL/3 and put it into `pretrained/PGDiT_200000_mask94_ema.pth`

## Training and Sampling
**Training:**  
```bash
python -m train.py --config hcp_dit.
```
you may adjust model size, patch size and buffered DiT layers in the `model/models.py `according to your training environment. 

**Sampling:**  
```bash
python -m sample.py --config hcp_dit.
```

## Acknowledgement
Our codebase is built based on DiT, VDT, MVCD, and Q-Former. We thank the authors for their nice code!
