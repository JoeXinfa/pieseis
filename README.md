# Python implementation of JavaSeis
This is a python port of the seismic file format named 'JavaSeis'.


## Why Python?
Because it's awesome!
And its awesome to use within scientific computing with the wonderful packages such as numpy and scipy.


## Distutils package
Will create and maintain a a python egg installer or tar.gz package with a corresponding setup.py to automate the installation.


## Example read 
```python
import pyjavaseis.io.jsfile as jsfile


PATH_TO_JS_DATASET = '/data/my_dataset.js'
js_reader = jsfile.JSFileReader()
js_reader.open(PATH_TO_JS_DATASET)

dataset = js_reader.dataset
file_properties = dataset.file_properties

print("Dataset {0}".format(PATH_TO_JS_DATASET))
print("Samples: {0}, Traces: {1}, Frames: {2}".format(
    js_reader.nr_samples,
    js_reader.nr_traces,
    js_reader.nr_frames))

```

---

## Author
Name: Asbjørn Alexander Fellinghaug
