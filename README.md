# RIDERS: Radar-Infrared Depth Estimation for Robust Sensing

## Background
Dense depth recovery is crucial in autonomous driving, serving as a foundational element for obstacle avoidance, 3D object detection, and local path planning. Adverse weather conditions, including haze, dust, rain, snow, and darkness, introduce significant challenges to accurate dense depth estimation, thereby posing substantial safety risks in autonomous driving. These challenges are particularly pronounced for traditional depth estimation methods that rely on short electromagnetic wave sensors, such as visible spectrum cameras and near-infrared LiDAR, due to their susceptibility to diffraction noise and occlusion in such environments.
To fundamentally overcome this issue, we present a novel approach for robust metric depth estimation by fusing a millimeter-wave Radar and a monocular infrared thermal camera, which are capable of penetrating atmospheric particles and unaffected by lighting conditions.
## Pipeline
Our proposed Radar-Infrared fusion method achieves highly accurate and finely detailed dense depth estimation through three stages, including monocular depth prediction with global scale alignment, quasi-dense Radar augmentation by learning Radar-pixels correspondences, and local scale refinement of dense depth using a scale map learner. Our method achieves exceptional visual quality and accurate metric estimation by addressing the challenges of ambiguity and misalignment that arise from directly fusing multi-modal long-wave features. 
## Demo
https://github.com/MMOCKING/RIDERS/assets/61079012/880ed67b-4fe1-4452-af91-e4f20c7f5b3a
## ZJU-Multispectrum Dataset
Download link: [ZJU-Multispectrum](https://pan.baidu.com/s/1TGPGjX8XtQf1CyKMqU2v_w?pwd=1897).
Code for extracting netdisk data if needed: 1897

Download link (APRIL LAB): [ZJU-Multispectrum](http://gofile.me/5oQXF/eaDrbYcaV).

```
ZJU-Multispectrum
├── [sequence names]
│   ├── any  # monocular depth predictions from Depth Anything (thermal images)
│   ├── lidar_png  # sparse lidar depths projected onto thermal images
│   ├── lidar_png_int  # interpolated lidar depths projected onto thermal images
│   ├── rgb_sync # RGB images
│   ├── thermal_undistort # thermal images
│   ├── radar_png # png files of radar depths projected onto thermal images
├── output
│   ├── rcnet_0.1
│   │   ├── depth_predicted # quasi-dense depth from RC-Net (output threshold = 0.1)
├── log # store training results
│   ├── rcnet # store .pth files
│   ├── SML # store .pth files
```


## Usage

Setup dependencies:

```
conda env create -f environment.yaml
conda activate rc-depth
```

Download [ZJU-Multispectrum](https://pan.baidu.com/s/1TGPGjX8XtQf1CyKMqU2v_w?pwd=1897) 
and use `val_zju.py` for quick starting.

Sequences for training and validation: 
```
['2023-10-19-19-25-47', '2023-10-20-10-05-18', '2023-10-20-10-21-14',
'2023-10-20-10-35-20', '2023-10-20-13-56-28', '2023-10-20-14-23-10', 
'2023-10-20-14-15-25', '2023-10-20-14-28-18', '2023-10-20-14-38-17', 
'2023-10-20-14-53-28']
```

Sequences for quantitative tests (clear day): 
```
['2023-10-20-10-07-22', '2023-10-20-10-28-46', '2023-10-20-14-35-31']
```

Other sequences are used for qualitative tests and robustness tests (smoke or low-light).


### Monocular Predictions
Depth Anything: https://github.com/LiheYoung/Depth-Anything

### RC-Net
For intermediate output (quasi-dense depth):

```
python train_rcnet_zju.py
python run_rcnet_zju.py
```

### Scale Map Learner
For final dense depth:

```
python train_zju.py
python val_zju.py
```

## Acknowledgements

Our work builds on and uses code from [Depth Anything](https://github.com/LiheYoung/Depth-Anything), 
[LeReS](https://github.com/aim-uofa/AdelaiDepth),
[ZoeDepth](https://github.com/isl-org/ZoeDepth),
[VI-Depth](https://github.com/isl-org/VI-Depth), 
and [radar-camera-fusion-depth](https://github.com/nesl/radar-camera-fusion-depth). 
We'd like to thank the authors for making these libraries and frameworks available.

