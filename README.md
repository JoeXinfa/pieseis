# PieSeis

PieSeis is a Python library for reading and writing JavaSeis dataset. It is built upon previous work in <a href=https://github.com/asbjorn/pyjavaseis>PyJavaSeis</a> and <a href=https://github.com/ChevronETC/TeaSeis.jl>TeaSeis.jl</a>.

## Maintainer

* Xinfa Joseph Zhu  <xinfazhu@gmail.com>

## Dependencies

* NumPy
* LXML

## Installation

* To use with a specific project, simply copy the pieseis subdirectory
    anywhere that is importable from your project.

* TODO To install system-wide from source distribution:
    `$ python setup.py install`

* TODO To install using a package management system:
    `pip install pieseis`
    `conda install pieseis`

## Example read

```python
import pieseis.io.jsfile as jsfile

# open to read
filename = 'C:/Users/joseph/Documents/181030_in.js'
jsd = jsfile.JavaSeisDataset.open(filename)
fps = jsd.file_properties

print("dataset name =", jsd.filename)
print("axis labels =", fps.axis_labels)
print("axis lengths =", fps.axis_lengths)
print("header entries =", len(jsd.trace_properties._trace_headers))
print("header bytes =", jsd.header_length)

# read trace header per trace
itrace, iframe = 1, 1
print("ILINE_NO =", jsd.get_trace_header("ILINE_NO", itrace, iframe)[0])
print("XLINE_NO =", jsd.get_trace_header("XLINE_NO", itrace, iframe)[0])

# read trace data per frame
iframe = 1
data = jsd.read_frame_trcs(iframe)
print("data shape =", data.shape)
print("first sample =", data[0, 0])
print("first trace =", data[0,:])
```

Look at `examples\read_file.py` for more examples.

## Example write

```python
import numpy as np
import pieseis.io.jsfile as jsfile

nsample, ntrace, nframe = 101, 101, 101
fold = ntrace # full fold

# open to write
filename = 'C:/Users/joseph/Documents/test/181030_out.js'
axis_lengths = [nsample, ntrace, nframe]
jsd = jsfile.JavaSeisDataset.open(filename, 'w', axis_lengths=axis_lengths)

# prepare trace data - 2D array for one frame
trcs = np.ones((ntrace, nsample))

# prepare header data
TFULL_S, TFULL_E = 0.0, 3000.0
TRC_TYPE = np.ones(ntrace, dtype='int32')
SEQNO = np.linspace(1, ntrace, ntrace, dtype='int32')
headers = {}
headers["TFULL_S"] = TFULL_S
headers["TFULL_E"] = TFULL_E
headers["TRC_TYPE"] = TRC_TYPE
headers["SEQNO"] = SEQNO

# write trace headers and data frame by frame
for i in range(nframe):
    iframe = i + 1
    headers["FRAME"] = iframe
    jsd.write_frame(trcs, headers, fold, iframe)
```

Look at `examples\write_file.py` for more examples.