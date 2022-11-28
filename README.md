<summary>Install</summary>

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).

```bash
pip install -r requirements.txt  # install
```

If you want to do a custom training, the differents steps should be : 
- 1 Create your dataset with the correct format and respecting the paths
- 2 Set up clear ml and optuna if necessary
- 3 Train, selecting the parameters required 

training_tuto.ipynb has everything you need to make a training, be shure to read all the points below to know more about the differents parameters. 

</details>

<details >
<summary>Inference</summary>

YOLOv5 [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36) inference. [Models](https://github.com/ultralytics/yolov5/tree/master/models) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases).

```python
import torch

# Model
model = torch.hub.load('your model')  # or yolov5n - yolov5x6, custom

# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
```

</details>

<details>
<summary>Inference with detect.py</summary>

`detect.py` runs inference on a variety of sources, downloading [models](https://github.com/ultralytics/yolov5/tree/master/models) automatically from
the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          screen  # screenshot
                          path/  # directory
                          'path/*.jpg'  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training on coco</summary>

The commands below reproduce YOLOv5 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)
results. [Models](https://github.com/ultralytics/yolov5/tree/master/models)
and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). 

```bash
python train.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                       yolov5s                                64
                                       yolov5m                                40
                                       yolov5l                                24
                                       yolov5x                                16
```

Just a good habit to test if everything is working correctly before going serious. 

</details>


<details >
<summary>Training on custom dataset</summary>

The commands below trains YOLOv5 on custom dataset. 

Train a YOLOv5 model on YOURDATASET by specifying dataset, batch-size, image size and either pretrained --weights yolov5s.pt (recommended), or randomly initialized --weights '' --cfg yolov5s.yaml (not recommended). Pretrained weights are auto-downloaded from the latest YOLOv5 release.

```bash
python train.py --data YOURDATASET.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                       yolov5s                                64
                                       yolov5m                                40
                                       yolov5l                                24
                                       yolov5x                                16
```

All training results are saved to runs/train/ with incrementing run directories, i.e. runs/train/exp2, runs/train/exp3 etc.


3loogers are avaible to folow your training, folow the init process (simply a connexion with your identifications) : 
```Python
#@title Select YOLOv5 🚀 logger {run: 'auto'}
logger = 'TensorBoard' #@param ['TensorBoard', 'Comet', 'ClearML']
if logger == 'TensorBoard':
  %load_ext tensorboard
  %tensorboard --logdir runs/train
elif logger == 'Comet':
  %pip install -q comet_ml
  import comet_ml; comet_ml.init()
elif logger == 'ClearML':
  %pip install -q clearml && clearml-init
```
DO IT BEFORE STARTING YOUR TRAINING OR IT WON'T WORK
</details>

<details >
<summary>Organising your dataset in yaml format</summary>

1. Create Dataset with yaml config, Becarfull that paths on the .yaml are respected. 
shown below, is the dataset config file that defines 1) the dataset root directory path and relative paths to train / val / test image directories (or *.txt files with image paths) and 2) a class names dictionary:

```c
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco128  # dataset root dir
train: images/train2017  # train images (relative to 'path') 128 images
val: images/train2017  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes (4 classes)
names:
  0: joueur
  1: ballon
  2: arbitre
  3: panier
```

2. Create Labels
After using a tool like Supervisly Annotate to label your images, export your labels to supervisly format.  
Use the script (on linux no windows) from the git that allow you to transform a supervisly format into a yolo format. 

It should looks like that :
![alt text](https://user-images.githubusercontent.com/26833433/112467037-d2568c00-8d66-11eb-8796-55402ac0d62f.png)
- One row per object
- Each row is class x_center y_center width height format.
- Box coordinates must be in normalized xywh format (from 0 - 1). If your boxes are in pixels, divide x_center and width by image width, and y_center and height by image height.
- Class numbers are zero-indexed (start from 0).


3. Organize Directories
Organize your train and val images and labels according to the example below. YOLOv5 assumes /'your dataset' is inside a /datasets directory next to the /'your dataset' directory. YOLOv5 locates labels automatically for each image by replacing the last instance of /images/ in each image path with /labels/. 


Becarfull, if you have a picture with double extension like aaa.jpg.jpg due to supervisely. The labels txt has to be aaa.jpg.txt, else it won't work. 
![alt text](https://user-images.githubusercontent.com/26833433/134436012-65111ad1-9541-4853-81a6-f19a3468b75f.png)

![alt text](https://i.gyazo.com/bf6a392c015ba9d9e8a373996792d716.png)
</details>

<details >
<summary>After training, where can you find the weights of your model?</summary>

A folder named run will be created after your first train. 
You will find a folder named 'exp' ('exp2' for a second training, 'exp3'...).
Inside a folder named weights, with your weights with the best training set-up, and the last one.

You can now do inference with those weights if needed. 

You will also find differents graphs concerning your training.  
</details>


<details >
<summary> Hyperparameter Optimization(for clearml and optuna)</summary>
To run hyperparameter optimization locally, we've included a pre-made script for you. Just make sure a training task has been run at least once, so it is in the ClearML experiment manager, we will essentially clone it and change its hyperparameters.

You'll need to fill in the ID of this template task in the script found at utils/loggers/clearml/hpo.py and then just run it :) You can change task.execute_locally() to task.execute() to put it in a ClearML queue and have a remote agent work on it instead.

```bash
# To use optuna, install it first, otherwise you can change the optimizer to just be RandomSearch
pip install optuna
python utils/loggers/clearml/hpo.py
```

</details>

<details >
<summary>How to Transfer Learning </summary>
Looking at the model architecture we can see that the model backbone is layers 0-9, so we can define the freeze list to contain all modules with 'model.0.' - 'model.9.' in their names.

The head with correct numbers of outputs will be automaticly generated with the .yaml file corresponding to your dataset. 

This is the command to use to train with only the backboned freezed
```bash
python train.py --freeze 10
```

To freeze the full model except for the final output convolution layers in Detect(), we set freeze list to contain all modules with 'model.0.' - 'model.23.' in their names:

```bash
python train.py --freeze 24
```

Results are way better if you only freeze the backbone, check the graphs at the end of this https://github.com/ultralytics/yolov5/issues/1314
There is a benchmark between a full freeze, only backbone freeze and the back bone + head freezed (expect the last layer)

</details>


[Tips for Best Training Results](https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results)