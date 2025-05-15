# SDF360
SDF360 is a method which reconstructs 3D data from 360-degree images

```
python main.py --mode train --conf. ./confs/womask.conf --case <case_name>
```
To extract mesh, run the commands below.

```
python main.py --mode validate_mesh --conf <config_file from exp> --case <case_name> --is_continue
```
