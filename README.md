# AST-Net: Modeling Spectro-Temporal Anisotropy for EEG-Based Seizure Detection 
This repository contains the official PyTorch implementation of **AST-Net** (Asymmetric Spectral-Temporal Network), a physiology-aware deep learning framework tailored for EEG-based seizure detection using 2D time-frequency scalograms. 
 ## Data Availability Statement 
 The multi-channel rat EEG dataset used in this study is non-public. To request access for academic or verification purposes, researchers must contact the original authors of the dataset. 
 For detailed information regarding the experimental cohorts, surgical procedures, and ethical approvals, please refer to the primary publication: * **Reference:** Yi, Y., Zhang, S., Dai, J., Zheng, H., Peng, X., Cheng, L., ... & Hu, Y. (2024). MiR-23b-3p improves brain damage after status epilepticus by reducing the formation of pathological high-frequency oscillations via inhibition of cx43 in rat hippocampus. _ACS Chemical Neuroscience_, _15_(14), 2633-2642.

---

## Repository Structure

```text
├── data/
│   ├── raw_data/                 # Place initial raw rat EEG storage data here
│   └── (Auto-generated intermediate data chunks will be directed here)
├── models/                       # Target directory for model checkpoints and logs
├── requirements.txt              # Environment dependencies and package specifications
├── .gitignore                    # Git tracking ignore file to safeguard massive datasets
├── dataset_prepare.py            # Phase 1: Data storage cleansing and anomaly removal
├── data_labeler.py               # Phase 2: Metadata indexing and labels.csv generation
├── channel_filter.py             # Phase 3: 8-channel filtration and 50Hz notch filter conditioning
├── data_segmentation.py          # Phase 4: Slicing continuous waveforms into 2s overlapping windows
├── time_frequency_transform.py   # Phase 5: Parallelized Continuous Wavelet Transform (CWT) scaling
└── ast_net_train.py              # Phase 6: Channel-independent optimization and soft voting ensemble
```
## Execution Pipeline

To fully replicate our preprocessing sequence and execute model benchmarking, execute the following steps sequentially. The pipeline handles data dependencies and directory creation automatically.
Step 0: Setup Environment and Dependencies
```
pip install -r requirements.txt
```

### Step 1: Raw Storage Cleansing

Isolate corrupted data streams and clear unreferenced directory nodes:
```
python dataset_prepare.py
```

### Step 2: Formulate Central Metadata Indexing
Query the directory structure to synthesize the mapping layer (`labels.csv`):
```
python data_labeler.py
```
### Step 3: Extract Effective Channels and Apply Filter
Filter the matrix down to the 8 clinically relevant EEG channels (6 electrodes in hippocampal subregions CA1, CA3, and dentate gyrus, plus 2 reference electrodes) and apply notch filtering to suppress 50Hz powerline interference:
```
python channel_filter.py
```
### Step 4: Temporal Window Slicing
Slice continuous temporal waveforms into discrete segments (`.npy` blocks) using 2-second windows with a 50% overlap ratio:

    python ast_net_train.py

## AST-Net Architectural Design

The core asymmetric engineering is implemented inside the `ASTConvBlock` and `ASTIdentityBlock` structures within `ast_net_train.py`. To resolve the anisotropic structural properties of CWT representations, feature maps are split into 4 concurrent transformation paths:

-   **Point Branch ($\mathcal{L}_{point}$):** A $1 \times 1$ convolution that preserves localized neural activity and channel-wise energy variations, serving as a stabilizing reference pathway.
    
-   **Isotropic Branch ($\mathcal{L}_{iso}$):** A standard $3 \times 3$ convolution that captures joint time-frequency interactions and complex patterns.
    
-   **Temporal Branch ($\mathcal{L}_{temp}$):** A slender $3 \times 11$ convolution kernel that emphasizes temporal continuity and models sustained oscillatory activity.
    
-   **Spectral Branch ($\mathcal{L}_{spec}$):** A vertical $7 \times 1$ convolution kernel that captures spectrally extended transients occurring over short temporal windows, characteristic of epileptic spikes.

![The whole framework](https://github.com/ZJM-ops/AST_Net__Modeling_Spectro_Temporal_Anisotropy_for_EEG_Based_Seizure_Detection/blob/main/framework.pdf)
## Training Hyperparameters
As specified in the paper, the network is trained from scratch with the following configuration:
-   **Hardware:** Single NVIDIA A100 GPU (40GB)
    
-   **Epochs:** 100
    
-   **Batch Size:** 32
    
-   **Optimizer:** SGD with momentum (0.9) and weight decay ($1 \times 10^{-4}$)
    
-   **Learning Rate Scheduler:** Cosine annealing with an initial learning rate of 0.01
