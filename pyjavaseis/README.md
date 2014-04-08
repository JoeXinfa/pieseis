pyjavaseis
==========

A python implementation of the JavaSeis format.

The purpose of this extension is the give users the ability to read and write to JavaSeis file formats, to ease the entrance of Python into the scientific computing area. Many python developers use well known scientific and mathematical python libraries such as "numpy", "scipy" and possibly also "matplotlib" to provide visualization.

Having the ability to extract the raw data and it's corresponding meta data is essential - therefore I started this python extension - pyjavaseis


## Development / extending pyjavaseis?
Change the DEBUG flag in <code>utils/config.py</code> to <code>True</code> to get debug output into the log file.

For normal usage I recommend setting this flag to <code>False</code>.


## Howto read a float and not double
By using the ```import struct``` module one can use floats without allocation a double in the native code. For instance: 
```python
import struct
struct.unpack("f", struct.pack("f", 0.05023324454))
```

The challenge would be to do this in a Pythonic way and using threading without being restricted by the GIL..



Author(s)
=========
* Asbjørn Alexander Fellinghaug (asbjorn ,dot, fellinghaug _dot_ com)
