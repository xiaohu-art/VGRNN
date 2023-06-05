# VGRNN implementation in Python 3.8 on cuda 11.3

The idea to implement such a version is because the [office implementation](https://github.com/VGraphRNN/VGRNN) is not compatible with the latest version of python library and cuda.

at begin we need to install the python library:
```bash
pip install -r requirements.txt
```

for basic exeucation:
```bash
python detection.py conv_type=SAGE datasets=fb
```

and we can get a dir `./output` and a picture named by `[task]_[conv_type]_[datasets].png` which records the training process and the final result of the corresponding config 
