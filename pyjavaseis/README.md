pyjavaseis
==========

A python implementation of the JavaSeis format.

The purpose of this extension is the give users the ability to read and write to JavaSeis file formats, to ease the entrance of Python into the scientific computing area. Many python developers use well known scientific and mathematical python libraries such as "numpy", "scipy" and possibly also "matplotlib" to provide visualization.

Having the ability to extract the raw data and it's corresponding meta data is essential - therefore I started this python extension - pyjavaseis


## Howto read a float and not double
By using the <code>import struct</code> module one can use floats without allocation a double in the native code. For instance: 
<code>
    import struct
    struct.unpack("f", struct.pack("f", 0.05023324454))
</code>

The challenge would be to do this in a Pythonic way and using threading without being restricted by the GIL..



Author(s)
=========
* Asbjørn Alexander Fellinghaug (asbjorn ,dot, fellinghaug _dot_ com)
