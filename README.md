# Regressive Guidance: A Framework to Guide the Spatial-Temporal Forecasting
#### *by: Jianyang Qin, Yan Jia, Binxing Fang and Qing Liao*


## Requirements:
- python >= 3.7
- pytorch >= 1.9.0
## Datasets
You can download the preprocessed datasets from [BaiduYun](https://pan.baidu.com/s/1pnFtk6G4wRo-DoES4o7_BA?pwd=vuxb), password: vuxb.


We used six public datasets in this study:
- HZ-METRO
- SH-METRO
- PEMS04
- PEMS08
- Wind
- Era5

## Usage 
You can select one of several training modes:
 - Create "exps" folder

 - Download the datasets and put them in "datasets" folder

 - Run with "python main.py --config configs/HZ-METRO/HZMETRO_astgcn.yaml --gpu 0" for model astgcn on the HZ-METRO dataset using one gpu 0

 - Run with "python main.py --config configs/HZ-METRO/HZMETRO_astgcn.yaml --gpu 0,1 --use_multi_gpu" for model astgcn on the HZ-METRO dataset using multiple gpus 0 and 1

 - Run with "sh run_hzmetro.sh" for all models on the HZ-METRO dataset

   ```
   python main.py --config configs/HZ-METRO/HZMETRO_astgcn.yaml --gpu 0
   ```

   ```
   python main.py --config configs/HZ-METRO/HZMETRO_astgcn.yaml --gpu 0,1 --use_multi_gpu
   ```
   
   ```
   sh run_hzmetro.sh
   ```

 - Check the output results (MAE and RMSE). Models are saved to "exps" folder for further use.
