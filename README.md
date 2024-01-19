# test-on-edge-device

### code copied from repo tensorflow-yolov4-tflite by hunglc007

<b> link to the original <a href="https://github.com/hunglc007/tensorflow-yolov4-tflite">repo</a> </b>

<hr>

The `/checkpoints` folder has two `.tflite` models(fp-32 and fp-16 quantized) for object detection, trained on clsses <b>ash</b>, <b>fire</b> and <b>smoke</b> on Yolov4-tiny and then converted to Tensorflow-Lite equivalent for deployment on edge devices. Also /data/classes/coco.names was updated accordingly with model needs.

### Demo

```bash
# Run demo tflite model (image)
python detect.py --weights  ./checkpoints/yolov4-416-fp16.tflite --size 416 --model yolov4 --image ./data/AFL/img10.jpg --framework tflite

# Run demo tflite model (video)
 python detectvideo.py --weights  ./checkpoints/yolov4-416-fp16.tflite --size 416 --model yolov4 --video ./data/AFL/video1.mp4 --framework tflite
```

<br>

### Output

##### Yolov4-tiny original weight
Inference time: ~= 200 ms (CPU: AMD Ryzen 5000 [Plugged In]) 
<p align="center"><img src="data/yolov4tiny.jpg" width="640"\></p>

<br>

##### Yolov4-tiny to tflite-fp32 converted model
Inference time: ~= 140 ms - 145 ms (CPU: AMD Ryzen 5000 [Plugged In]) 
<p align="center"><img src="data/tflitefp32.png" width="640"\></p>

<br>

##### Yolov4-tiny to tflite-fp16 converted model
Inference time: ~= 110 ms - 120 ms (CPU: AMD Ryzen 5000 [Plugged In]) 
<p align="center"><img src="data/tflitefp16.png" width="640"\></p>

<hr>

### Install Requirements:

> pip install -r requirements.txt

<hr>