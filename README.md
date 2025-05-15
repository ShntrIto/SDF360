# SDF360
SDF360 is a method which reconstructs 3D mesh from 360-degree images

1. Clone this repository

```
git clone https://github.com/ShntrIto/SDF360.git
cd SDF360
```

2. Preprocess data
To train NeuS, it is necessary to preprocess the dataset according to the instructions provided in the [Training NeuS Using Your Custom Data](https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data)

3. Run SDF360

```
python main.py --mode train --conf. ./confs/womask_erp.conf --case demo-data
```

4. Extract mesh

```
python main.py --mode validate_mesh --conf ./confs/womask_erp.conf --case demo-data --is_continue
```   
5. Evaluate mesh
Please refer "Example of 3D Reconstruction with NeuS" in the GitHub repository below to evaluate 3D mesh.

https://github.com/gsisaoki/Omnidirectional_Blender_3D_Dataset

# OB3D Dataset
Please download our dataset through [this link](https://www.kaggle.com/datasets/shintacs/ob3d-dataset).
