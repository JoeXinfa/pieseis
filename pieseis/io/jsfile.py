import os
import os.path as osp
import struct
import warnings

from lxml import etree
import numpy as np

from .properties import (FileProperties, TraceProperties, CustomProperties,
    VirtualFolders, TraceFileXML, TraceHeadersXML, TraceHeader)
from .defs import GridDefinition
from .trace_compressor import get_trace_compressor, get_trace_length, unpack_frame
from .compat import dictJStoPM
from .stock_props import minimal_props, stock_props

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
    if osp.exists(path):
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
    def __init__(self, filename):
#        if not osp.isdir(filename):
#            self.is_valid = True
#            raise IOError("Must be a folder: %s" % filename)
#
#        try:
#            self._validate_js_dir(filename)
#            self._is_valid = True
#        except IOError as ioe:
#            print("%s is not a valid dataset" % filename)
#            print("msg: %s" % ioe)
#            self._is_valid = False

        #self._files = os.listdir(filename)
        self.path = filename

        # self.read_data()
        #self._is_open = True

    @classmethod
    def open(cls, filename,
        mode                = 'r',
        description         = "",
        mapped              = True,
        data_type           = None,
        data_format         = "float32",
        data_order          = "",
        axis_lengths        = [],
        axis_propdefs       = {},
        axis_units          = [],
        axis_domains        = [],
        axis_lstarts        = [],
        axis_lincs          = [],
        axis_pstarts        = [],
        axis_pincs          = [],
        data_properties     = [],
        properties          = {},
        geometry            = None,
        secondaries         = None,
        nextents            = 0,
        similar_to          = "",
        properties_add      = {},
        properties_rm       = {},
        data_properties_add = [],
        data_properties_rm  = []):
        """
        -i- filename : string, path of the JavaSeis dataset e.g. "data.js"
        -i- mode : string, "r" read, "w" write/create, "r+" read and write
        Utility method to open a JavaSeis dataset. Will throw an
        IOError exception in case of any errors, or a
        JavaSeisDataset instance.
        """
        jsd = JavaSeisDataset(filename)
        if mode == 'r' or mode == 'r+':
            if not osp.isdir(filename):
                # TODO: call create_javaseis(filename) here
                raise IOError("JavaSeis dataset does not exists")
            if not os.access(filename, os.R_OK):
                raise IOError("Missing read access for JavaSeis dataset")

            jsd._read_properties()
            jsd._set_trace_format()

            trace_properties = jsd._trace_properties._trace_headers
            axis_labels = jsd._file_properties.axis_labels
            _axis_propdefs = get_axis_propdefs(trace_properties, axis_labels)
            _axis_lengths = jsd._file_properties.axis_lengths
            _data_format = jsd._trace_format

        elif mode == 'w' and similar_to == "":
            ndim = len(axis_lengths)
            if ndim < 2:
                raise ValueError("axis dimension must >= 2")
            trace_properties, _axis_propdefs = get_trace_properties(ndim,
                properties, properties_add, properties_rm, axis_propdefs,
                similar_to)
            _axis_lengths = axis_lengths
            _data_format = data_format

        #print('trace_properties =', trace_properties)
        #print('_axis_propdefs =', _axis_propdefs)
        for key, th in trace_properties.items():
            print(key, th.label, th.format, th.element_count, th.byte_offset)
        for key, th in _axis_propdefs.items():
            print(key, th.label, th.format, th.element_count, th.byte_offset)
        print('_axis_lengths =', _axis_lengths)
        print('_data_format =', _data_format)
        exit()

        jsd._set_trace_compressor()
        jsd.properties = trace_properties
        jsd.axis_propdefs = _axis_propdefs
        jsd.compressor = jsd._trace_compressor
        jsd.filename = filename
        jsd.mode = mode
        jsd.current_volume = -1
        jsd.header_length = jsd._trace_properties._total_bytes

        if mode == 'r' or mode == 'r+':
            filename = osp.join(jsd.path, JS_FILE_STUB)
            jsd.description = get_description(filename)

            jsd.mapped          = jsd._file_properties.is_mapped()
            jsd.data_type       = jsd._file_properties.data_type
            jsd.data_format     = jsd._trace_format
            jsd.data_order      = jsd._file_properties.byte_order
            jsd.axis_lengths    = jsd._file_properties.axis_lengths
            jsd.axis_units      = jsd._file_properties.axis_units
            jsd.axis_domains    = jsd._file_properties.axis_domains
            jsd.axis_lstarts    = jsd._file_properties.logical_origins
            jsd.axis_lincs      = jsd._file_properties.logical_deltas
            jsd.axis_pstarts    = jsd._file_properties.physical_origins
            jsd.axis_pincs      = jsd._file_properties.physical_deltas
            jsd.data_properties = jsd._custom_properties.data_properties
            jsd.geom            = None

            jsd.has_traces = False
            filename = osp.join(jsd.path, JS_HAS_TRACES_FILE)
            if osp.isfile(filename):
                jsd.has_traces = get_status(filename)

            jsd._read_virtual_folders()
            jsd.secondaries = jsd._virtual_folders.secondary_folders

            jsd._read_trace_file_xml()
            jsd._set_trc_extents()
            jsd.trc_extents = jsd._trc_extents

            jsd._read_trace_headers_xml()
            jsd._set_hdr_extents()
            jsd.hdr_extents = jsd._hdr_extents

            jsd.current_volume = -1
            jsd.map = np.zeros(jsd.axis_lengths[2], dtype='int32')

            #jsd._set_trace_format()
            #jsd._set_trace_compressor()
            #jsd._set_trace_length()

            return jsd

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

    def _read_trace_headers_xml(self):
        filename = osp.join(self.path, JS_TRACE_HEADERS_XML)
        root = parse_xml_file(filename)
        self._trace_headers_xml = TraceHeadersXML(root)

    def _read_trace_file_xml(self):
        filename = osp.join(self.path, JS_TRACE_DATA_XML)
        root = parse_xml_file(filename)
        self._trace_file_xml = TraceFileXML(root)

    def _read_virtual_folders(self):
        filename = osp.join(self.path, JS_VIRTUAL_FOLDERS_XML)
        root = parse_xml_file(filename)
        self._virtual_folders = VirtualFolders(root)

    def _read_properties(self):
        filename = osp.join(self.path, JS_FILE_PROPERTIES_XML)
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
        self._trace_properties = TraceProperties(parset_trace_properties)
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

    @file_properties.setter
    def file_properties(self, x):
        self._file_properties = x

    @property
    def trace_properties(self):
        return self._trace_properties

    @trace_properties.setter
    def trace_properties(self, x):
        self._trace_properties = x

    @property
    def custom_properties(self):
        return self._custom_properties

    @custom_properties.setter
    def custom_properties(self, x):
        self._custom_properties = x

    def __str__(self):
        return "<JavaSeisDataset {}>" \
            .format(self.path)

    def _set_trace_format(self):
        trcfmt = self.file_properties.trace_format
        if trcfmt == "FLOAT":
            self._trace_format = "float32"
        elif trcfmt == "DOUBLE":
            self._trace_format = "float64"
        elif trcfmt == "COMPRESSED_INT32":
            self._trace_format = "int32"
        elif trcfmt == "COMPRESSED_INT16":
            self._trace_format = "int16"
        else:
            raise ValueError("Unrecognized trace format".format(trcfmt))

    def _set_trace_compressor(self):
        nsamples = self.file_properties.axis_lengths[0]
        trace_format = self._trace_format
        self._trace_compressor = get_trace_compressor(nsamples, trace_format)

    def _set_trace_length(self):
        compressor = self._trace_compressor
        self._trace_length = get_trace_length(compressor)

    def _set_trc_extents(self):
        xml = self._trace_file_xml
        secondaries = self._virtual_folders.secondary_folders
        filename = self.path
        self._trc_extents = get_extents(xml, secondaries, filename)

    def _set_hdr_extents(self):
        xml = self._trace_headers_xml
        secondaries = self._virtual_folders.secondary_folders
        filename = self.path
        self._hdr_extents = get_extents(xml, secondaries, filename)

    def _get_fold(self, iframe):
        # TODO read TraceMap
        return self._file_properties.axis_lengths[1]

    @property
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
        self._js_dataset = JavaSeisDataset.open(path)

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
            trace_properties.total_bytes
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
            tracemap_file = osp.join(path, JS_TRACE_MAP)
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
        -o- trcs : array, float32, shape (ntrace, nsample)
        """
        trclen = self._js_dataset.trace_length
        fold = self._js_dataset._get_fold(iframe)

        #size = trclen * self.nr_traces # frame byte size
        offset = (iframe - 1) * trclen * self.nr_traces
        extents = self._js_dataset._trc_extents
        extent = get_extent_index(extents, offset)
        offset -= extent['start']
        trcfmt = self._js_dataset._trace_format
        if trcfmt == "int16":
            f = open(extent['path'], "rb")
            cps = self._js_dataset._trace_compressor
            array = unpack_frame(f, offset, cps, fold)
            return array
        elif trcfmt == "float32":
            pass
        else:
            raise ValueError("Unsupported trace format".format(trcfmt))

    def read_frame_hdrs(self):
        pass


def parse_xml_file(filename):
    with open(filename, 'r') as f:
        data = f.read()
    root = etree.XML(data)
    return root


def extent_dir(secondary, filename):
    """
    -i- secondary : string, path of JavaSeis secondary disk
    -i- filename : string, JavaSeis dataset name
    -o- extdir : string, the JavaSeis dataset secondary directory
    """
    isrelative = osp.isabs(filename) == False
    pmdh = "PROMAX_DATA_HOME"
    jsdh = "JAVASEIS_DATA_HOME"
    datahome = os.environ[pmdh] if pmdh in os.environ else ""
    datahome = os.environ[jsdh] if jsdh in os.environ else datahome
    if secondary == ".":
        return osp.abspath(filename)
    elif datahome != "":
        filename = osp.abspath(filename)
        if datahome in filename:
            # TODO test with Windows paths
            return filename.replace(datahome, secondary)
        message1 = "JAVASEIS_DATA_HOME or PROMAX_DATA_HOME is set, and " +\
        "JavaSeis filename is relative, but the working directory is not " +\
        "consistent with JAVASEIS_DATA_HOME: datahome={}, filename={}. " +\
        "Either unset JAVASEIS_DATA_HOME and PROMAX_DATA_HOME, " +\
        "make your working directory correspond to datahome, " +\
        "or use absolute file paths.".format(datahome, filename)
        message2 = "JAVASEIS_DATA_HOME or PROMAX_DATA_HOME is set " +\
        "but does not seem correct: " +\
        "datahome={}, filename={}".format(datahome, filename)
        if isrelative and not os.getcwd().startswith(datahome):
            raise ValueError(message1)
        raise ValueError(message2)
    elif isrelative:
        return osp.join(secondary, filename)
    else:
        pass # TODO joinpath(secondary, is_windows() ? filename[5:end] : filename[2:end])


def get_extent_index(extents, offset):
    i = int(offset / extents[0]['size'])
    return extents[i]

def get_extents(xml, secondaries, filename):
    """
    -i- xml : object, parsed XML file
    -i- secondaries : list, element is string of secondary folder path
    -i- filename : string, JavaSeis dataset filename
    """
    nextents = xml.nr_extents
    basename = xml.extent_name
    size = xml.extent_size
    maxpos = xml.extent_maxpos
    extents = [None] * nextents
    for secondary in secondaries:
        base_extpath = extent_dir(secondary, filename)
        if osp.isdir(base_extpath):
            names = get_names(base_extpath, basename)
            for name in names:
                i = int(name[len(basename):])
                if i < nextents:
                    start = i * size
                    path = osp.join(base_extpath, name)
                    extents[i] = make_extent(name, path, i, start, size)

    # TODO add missing extents (i.e. extents with all empty frames)

    # the last extent might be a different size
    extent = extents[nextents-1]
    extent['size'] = maxpos - extent['start']

    return extents


def make_extent(name, path, index, start, size):
    return {'name': name, 'path': path, 'index': index, 'start': start,
            'size': size}


def get_names(path, start):
    names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(start) and not file.endswith('.xml'):
                names.append(file)
    return names


def get_axis_propdefs(trace_headers, axis_labels):
    axis_propdefs = {}
    for i, label in enumerate(axis_labels):
        axis_propdefs[label] = get_axis_propdef(trace_headers, label, i+1)
    return axis_propdefs


def get_axis_propdef(trace_headers, label, dim):
    # map from JavaSeis axis name to ProMax property label
    plabel = dictJStoPM[label] if label in dictJStoPM else label

    for key, value in trace_headers.items():
        if value.label == plabel:
            return value

    # The sample and trace labels do not need a corresponding trace property.
    # Therefore, these should be considered valid datasets.
    if dim == 1 or dim == 2:
        return TraceHeader(values=(label, label, 'int32', 1, 0))

    raise ValueError("Malformed JavaSeis: axis props, axis label={} has no "
                     "corresponding trace property".format(label))


def get_description(filename):
    with open(filename, 'r') as f:
        for line in f:
            if line[0] != '#':
                line = line.strip()
                columns = line.split('=')
                if columns[0] == "DescriptiveName":
                    return columns[1]
    return ""


def get_status(filename):
    with open(filename, 'r') as f:
        for line in f:
            if line[0] != '#':
                line = line.strip()
                columns = line.split('=')
                if len(columns) < 2:
                    warnings.warn("Status info (has traces) may be incorrect")
                    return False
                if columns[0] == "HasTraces" and columns[1] == 'true':
                    return True
    return False


def get_trace_properties(ndim, property_defs, property_defs_add,
    property_defs_rm, axis_propdefs, similar_to):
    """
    -i- ndim : int, number of dimensions
    -i- property_defs : dict
    -i- property_defs_add : dict
    -i- property_defs_rm : dict
    -i- axis_propdefs : dict
    -i- similar_to : string
    'defs' is short for 'defaults' or 'defined'.
    """
    if similar_to == "":
        _property_defs = property_defs # need copy.deepcopy?
        _axis_propdefs = axis_propdefs
    else:
        jsd = JavaSeisDataset.open(similar_to, mode='r')

        # special handling for trace properties
        if len(property_defs) == 0:
            _property_defs = jsd.properties
        else:
            assert len(property_defs_add) == 0
            assert len(property_defs_rm) == 0

        for key in property_defs_add:
            if key not in _property_defs:
                _property_defs[key] = property_defs_add[key]
        for key in property_defs_rm:
            if key in _property_defs:
                _property_defs.pop(key)

        if len(axis_propdefs) == 0:
            _axis_propdefs = jsd.axis_propdefs
        else:
            _axis_propdefs = axis_propdefs

    # initialize trace properties to an empty dictionary
    properties = {}

    # trace properties, minimal set (as defined by SeisSpace / ProMAX)
    if similar_to == "":
        byte_offset = get_minimal_propset(properties, start_offset=0)
    else:
        byte_offset = 0

    # trace properties, user defined
    for key in _property_defs:
        if key not in properties:
            th = _property_defs[key]
            th.byte_offset = byte_offset
            properties[key] = th
            byte_offset += th.size

    # axis properties
    if len(_axis_propdefs) == 0:
        th_list = [stock_props['SAMPLE'], stock_props['TRACE'],
            stock_props['FRAME'], stock_props['VOLUME'],
            stock_props['HYPRCUBE']][:min(5,ndim)]
        for idim in range(6, ndim+1):
            label = "DIM{}".format(idim)
            description = "dimension {}".format(idim)
            th = TraceHeader(values=(label, description, "INTEGER", 1, 0))
            th_list.append(th)
        for th in th_list: # create dict from list
            _axis_propdefs[th.label] = th

    for key, th in _axis_propdefs.items():
        assert th.element_count == 1
        # map from JavaSeis axis name to ProMax property label
        if th.label in dictJStoPM:
            th.label = dictJStoPM[th.label]
        if th.label not in properties:
            th.byte_offset = byte_offset
            properties[th.label] = th
            byte_offset += th.size

    return properties, _axis_propdefs


def get_minimal_propset(properties, start_offset=0):
    """
    -i- properties : dict, get/set minimal trace properties and add to it
    -i- start_offset : int, start byte offset
    """
    byte_offset = start_offset
    for prop in minimal_props:
        th = stock_props[prop]
        th.byte_offset = byte_offset
        properties[th.label] = th
        byte_offset += th.size
    return byte_offset


if __name__ == '__main__':
    testpath = "/home/asbjorn/datasets/2hots.js"
    if not osp.exists(testpath):
        print("'{0}' dataset does not exists..".format(testpath))
    jsDataset = JavaSeisDataset(testpath)
    print(jsDataset)
