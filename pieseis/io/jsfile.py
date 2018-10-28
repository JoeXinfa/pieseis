import os
from lxml import etree
import struct

from .properties import (FileProperties, TraceProperties, CustomProperties,
    VirtualFolders)
from .defs import GridDefinition
from .trace_compressor import get_trace_compressor, get_trace_length

# constant filenames
JS_FILE_PROPERTIES_XML = "FileProperties.xml"
JS_FILE_PROPERTIES_OBS = "FileProperties"
JS_FILE_STUB = "Name.properties"
JS_TRACE_DATA_XML = "TraceFile.xml"
JS_TRACE_DATA = "TraceFile"
JS_TRACE_HEADERS = "TraceHeaders"
JS_HISTORY_XML = "History.xml"
JS_TRACE_MAP = "TraceMap"
JS_HAS_TRACES_FILE = "Status.properties"
JS_VIRTUAL_FOLDERS_XML = "VirtualFolders.xml"
JS_TRACE_HEADERS_XML = "TraceHeaders.xml"

# other constants
JS_EARLYVERSION1 = "2006.01"
JS_PREXMLVERSION = "2006.2"
JS_VERSION = "2006.3"


def create_javaseis(path):
    """
    Utility method to write / construct a javaseis dataset.
    Will either throw a IOError exception or a
    JavaSeisDataset instance.
    """
    if os.path.exists(path):
        raise IOError("Path for JavaSeis dataset already exists..")

    if not path.endswith(".js"):
        path += ".js"

    # Construct a new "JavaSeis" file here.
    try:
        os.makedirs(path)
    except IOError as e:
        raise e

    # TODO: Construct all the meta files
    raise NotImplementedError("Not yet fully implemented " +
                              "feature to create new JS datasets.")


class JavaSeisDataset(object):
    """
    Class to host a JavaSeis dataset. This class will be used by both the
    Reader and Writer classes.
    """
    def __init__(self, path):
        if not os.path.isdir(path):
            self.is_valid = True
            raise IOError("Must be a folder: %s" % path)

        try:
            self._validate_js_dir(path)
            self._is_valid = True
        except IOError as ioe:
            print("%s is not a valid dataset" % path)
            print("msg: %s" % ioe)
            self._is_valid = False

        self._files = os.listdir(path)
        self.path = path
        self._read_properties()
        self._read_virtual_folders()

        self._set_trace_format()
        self._set_trace_compressor()
        self._set_trace_length()

        # self.read_data()
        self._is_open = True

    @classmethod
    def open_javaseis(cls, path):
        """
        Utility method to open a JavaSeis dataset. Will throw an
        IOError exception in case of any errors, or a
        JavaSeisDataset instance.
        """
        if not os.path.isdir(path):
            # TODO: call create_javaseis(path) here
            raise IOError("JavaSeis dataset does not exists")

        if not os.access(path, os.R_OK):
            raise IOError("Missing read access for JavaSeis dataset")

        js_dataset = JavaSeisDataset(path)
        return js_dataset

    def _validate_js_dir(self, path):
        """Gets called during the construction of this object instance"""
        def js_missing(f):
            raise IOError("Missing: "+f)
        files = os.listdir(path)

        if JS_FILE_PROPERTIES_XML not in files:
            js_missing(JS_FILE_PROPERTIES_XML)

        if JS_HISTORY_XML not in files:
            js_missing(JS_HISTORY_XML)

        if JS_TRACE_DATA_XML not in files:
            js_missing(JS_TRACE_DATA_XML)

        if JS_TRACE_HEADERS_XML not in files:
            js_missing(JS_TRACE_HEADERS_XML)

        if JS_FILE_STUB not in files:
            js_missing(JS_FILE_STUB)

        if JS_HAS_TRACES_FILE not in files:
            js_missing(JS_HAS_TRACES_FILE)

        return True

    def _read_virtual_folders(self):
        filename = os.path.join(self.path, JS_VIRTUAL_FOLDERS_XML)
        root = parse_xml_file(filename)
        self._virtual_folders = VirtualFolders(root)

    def _read_properties(self):
        filename = os.path.join(self.path, JS_FILE_PROPERTIES_XML)
        root = parse_xml_file(filename)

        if root.get('name') != 'JavaSeis Metadata':
            raise IOError(JS_FILE_PROPERTIES_XML +
                          " is not a JavaSeis Metadata file!")

        children_nodes = list(root)

        def get_parset(name):
            tmp = [x for x in children_nodes if x.get('name') == name]
            if tmp and len(tmp) == 1:
                return tmp[0]
            else:
                return None

        parset_file_properties = get_parset('FileProperties')
        parset_trace_properties = get_parset('TraceProperties')
        parset_custom_properties = get_parset('CustomProperties')

        self._file_properties = FileProperties(parset_file_properties)
        #print(self._file_properties)
        #print(self._file_properties._attributes)

        self._trace_properties = TraceProperties(parset_trace_properties)
        #print(self._trace_properties)

        self._custom_properties = CustomProperties(parset_custom_properties)

    def is_open(self):
        return self._is_open

    def close(self):
        """
        Close any open file descriptors or data resources..
        """
        if self.is_open():
            return True
        return False

    def is_valid(self):
        return self._is_valid

    @property
    def virtual_folders(self):
        return self._virtual_folders

    @property
    def file_properties(self):
        return self._file_properties

    @property
    def trace_properties(self):
        return self._trace_properties

    @property
    def custom_properties(self):
        return self._custom_properties

    def __str__(self):
        return "<JavaSeisDataset {}>" \
            .format(self.path)

    def _set_trace_format(self):
        trcfmt = self.file_properties.trace_format
        if trcfmt == "FLOAT":
            self._trace_format = "Float32"
        elif trcfmt == "DOUBLE":
            self._trace_format = "Float64"
        elif trcfmt == "COMPRESSED_INT32":
            self._trace_format = "Int32"
        elif trcfmt == "COMPRESSED_INT16":
            self._trace_format = "Int16"
        else:
            raise ValueError("Unrecognized trace format".format(trcfmt))

    def _set_trace_compressor(self):
        nsamples = self.file_properties.axis_lengths[0]
        trace_format = self._trace_format
        self._trace_compressor = get_trace_compressor(nsamples, trace_format)

    def _set_trace_length(self):
        compressor = self._trace_compressor
        trace_format = self._trace_format
        self._trace_length = get_trace_length(compressor, trace_format)

    def trace_length(self):
        return self._trace_length


class JSFileReader(object):
    def __init__(self, bufferSize=2097152):
        self._bufferSize = bufferSize
        self._traceBuffer = None
        self._headerBuffer = None
        self._traceMap = None
        self._js_dataset = None
        self._num_samples = self._num_traces = self._num_volumes = None

    def open(self, path, nthreads=2):
        self._js_dataset = JavaSeisDataset.open_javaseis(path)

        props = self._js_dataset.file_properties

        self._num_samples = props.axis_lengths[GridDefinition.SAMPLE_INDEX]
        self._num_traces = props.axis_lengths[GridDefinition.TRACE_INDEX]
        if len(props.axis_lengths) > GridDefinition.FRAME_INDEX:
            self._num_frames = props.axis_lengths[GridDefinition.FRAME_INDEX]
        if len(props.axis_lengths) > GridDefinition.VOLUME_INDEX:
            self._num_volumes = props.axis_lengths[GridDefinition.VOLUME_INDEX]
        if len(props.axis_lengths) > GridDefinition.HYPERCUBE_INDEX:
            self._num_hypercubes = props.axis_lengths[GridDefinition.HYPERCUBE_INDEX]

        self._nthreads = nthreads
        self._header_length_in_bytes = self._js_dataset.\
            trace_properties.record_lengths
        self._frame_header_length = self._header_length_in_bytes * \
            self._num_traces

        self.is_regular = True

        # TODO: Use scipy / numpy to read the binary data?

        if self._js_dataset.file_properties.is_mapped():
            # read TraceMap and check whether it contains a
            # value that differs from m_numTraces.
            # If so, then it is not regular, otherwise regular
            print("Dataset is mapped!")
            self.is_mapped = True
            self.is_regular = True
            total_nr_of_live_traces = 0
            max_ints_to_read = 4096
            #int_buffer = [0] * max_ints_to_read
            nr_of_read_ints = 0

            #while nr_of_read_ints < self.total_nr_of_frames:
                #ints_to_read = self.total_nr_of_frames - nr_of_read_ints
                #if ints_to_read
            tracemap_file = os.path.join(path, JS_TRACE_MAP)
            print("Read binary data: {}".format(tracemap_file))

            with open(tracemap_file, 'rb') as f:
                ints2read = min(
                    (self.total_nr_of_frames - nr_of_read_ints),
                    max_ints_to_read)
                data = f.read(ints2read*4)  # int is 4 bytes
                unpacked_data = struct.unpack(
                    "={}".format("i"*ints2read), data)
                #print('tracemap type =', type(unpacked_data))
                #print('tracemap len =', len(unpacked_data))
                #print('tracemap data =', unpacked_data)
                for i in unpacked_data:
                    total_nr_of_live_traces += i
                    if i != self._num_traces:
                        print("i != num_traces: %i != %i" %
                             (i, self._num_traces))
                        print("Not regular..")
                        self.is_regular = False
        else:
            # if not mapped, then it must be regular
            self.is_mapped = False
            self.is_regular = True
            total_nr_of_live_traces = self.total_nr_of_traces

    @property
    def total_nr_of_frames(self):
        """Calculates the total number of frames in the dataset.
        Collect the number in the 3rd dimension (axis_lengths[3]), and also if
        we are a 4D dataset we also multiply this with the length of the 4th
        dimension.
        """
        nr_dim = self._js_dataset.file_properties.nr_dimensions
        total_frames = 1
        axis_lengths = self._js_dataset.file_properties.axis_lengths
        for dimension in range(2, nr_dim):
            total_frames *= axis_lengths[dimension]
        return total_frames

    @property
    def total_nr_of_traces(self):
        """Calculates the total number of traces in the dataset.
        """
        total_traces = self.nr_traces * self.total_nr_of_frames
        return total_traces

    @property
    def nr_samples(self):
        """Return the number of samples in the dataset"""
        return self._js_dataset.file_properties.axis_lengths[0]

    @property
    def nr_traces(self):
        """Return the number of traces in the dataset"""
        return self._js_dataset.file_properties.axis_lengths[1]

    @property
    def nr_frames(self):
        """Return the number of frames in the dataset"""
        return self._js_dataset.file_properties.axis_lengths[2]

    @property
    def dataset(self):
        return self._js_dataset

    @property
    def javaseis_dataset(self):
        return self._js_dataset

    def read_frame_trcs(self, iframe):
        """
        -i- iframe : int, index of frame, [1, nframe]
        -o- trcs : array, float32, shape (nsample, ntrace)
        """
        trclen = self._js_dataset.trace_length

        offset = (iframe - 1) * trclen * self.nr_traces
        #ext = extentindex(io.trcextents, offset)


    def read_frame_hdrs(self):
        pass


def parse_xml_file(filename):
    with open(filename, 'r') as f:
        data = f.read()
    root = etree.XML(data)
    return root


def get_extents():
    pass


if __name__ == '__main__':
    testpath = "/home/asbjorn/datasets/2hots.js"
    if not os.path.exists(testpath):
        print("'{0}' dataset does not exists..".format(testpath))
    jsDataset = JavaSeisDataset(testpath)
    print(jsDataset)
