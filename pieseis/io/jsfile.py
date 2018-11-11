import os
import os.path as osp
import struct
import warnings
import math
import shutil
import pytz
from datetime import datetime

from lxml import etree
import numpy as np

from .properties import (FileProperties, TraceProperties, CustomProperties,
    VirtualFolders, TraceFileXML, TraceHeadersXML, TraceHeader)
from .defs import GridDefinition
from .trace_compressor import get_trace_compressor, get_trace_length, unpack_frame
from .compat import dictJStoPM, dictPMtoJS
from .stock_props import (minimal_props, stock_props, stock_dtype,
                          stock_unit, stock_domain)

DOT_XML = ".xml"
JS_FILE_PROPERTIES = "FileProperties"
JS_TRACE_DATA = "TraceFile"
JS_TRACE_HEADERS = "TraceHeaders"
JS_VIRTUAL_FOLDERS = "VirtualFolders"
JS_HISTORY = "History"

# constant filenames
JS_TRACE_MAP = "TraceMap"
JS_NAME_FILE = "Name.properties"
JS_STATUS_FILE = "Status.properties"
JS_FILE_PROPERTIES_XML = JS_FILE_PROPERTIES + DOT_XML
JS_TRACE_DATA_XML = JS_TRACE_DATA + DOT_XML
JS_TRACE_HEADERS_XML = JS_TRACE_HEADERS + DOT_XML
JS_VIRTUAL_FOLDERS_XML = JS_VIRTUAL_FOLDERS + DOT_XML
JS_HISTORY_XML = JS_HISTORY + DOT_XML

# other constants
JS_EARLYVERSION1 = "2006.01"
JS_PREXMLVERSION = "2006.2"
JS_VERSION = "2006.3"

FORMAT_BYTE = {
    "float32" : 4,
    "float64" : 8,
    "int32"   : 4,
    "int16"   : 2
}

TRACE_FORMAT_TO_DATA_FORMAT = {
    "FLOAT"            : "float32",
    "DOUBLE"           : "float64",
    "COMPRESSED_INT32" : "int32",
    "COMPRESSED_INT16" : "int16"
}

DATA_FORMAT_TO_TRACE_FORMAT = {
    "float32" : "FLOAT",
    "float64" : "DOUBLE",
    "int32"   : "COMPRESSED_INT32",
    "int16"   : "COMPRESSED_INT16"
}


def check_io(filename, mode):
    if not filename.endswith(".js"):
        warnings.warn("JavaSeis filename does not end with '.js'")
    if mode == 'r' or mode == 'r+':
        if not osp.isdir(filename):
            raise IOError("JavaSeis filename is not directory")
        if not os.access(filename, os.R_OK):
            raise IOError("Missing read access")
    elif mode == 'w':
        if osp.exists(filename):
            raise IOError("Path for JavaSeis dataset already exists")
            # TODO give option to overwrite (delete old and write new)
        parent_directory = osp.dirname(filename)
        if not os.access(parent_directory, os.W_OK):
            raise IOError("Missing write access in {}".format(parent_directory))
    else:
        raise ValueError("unrecognized mode: {}".format(mode))


class JavaSeisDataset(object):
    """
    Class to host a JavaSeis dataset. This class will be used by both the
    Reader and Writer classes.
    """
    def __init__(self, filename):
        self.path = filename

    @classmethod
    def open(cls, filename,
        mode                = "r",
        description         = "",
        mapped              = None,
        data_type           = None,
        data_format         = None,
        data_order          = None,
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
        """
        check_io(filename, mode)
        jsd = JavaSeisDataset(filename)
        if mode == 'r' or mode == 'r+':
            jsd.is_valid = jsd._validate_js_dir(jsd.path)
            jsd._read_properties()
            trace_properties = jsd._trace_properties._trace_headers
            axis_labels = jsd._file_properties.axis_labels
            _axis_propdefs = get_axis_propdefs(trace_properties, axis_labels)
            _axis_lengths = jsd._file_properties.axis_lengths
            _data_format = TRACE_FORMAT_TO_DATA_FORMAT[jsd._file_properties.trace_format]
        elif mode == 'w' and similar_to == "":
            _axis_lengths = axis_lengths
            _data_format = "float32" if data_format is None else data_format
        elif mode == 'w' and similar_to != "":
            jsdsim = JavaSeisDataset.open(similar_to)
            _axis_lengths = jsdsim.axis_lengths if len(axis_lengths) == 0 else axis_lengths
            _data_format = jsdsim.data_format if data_format is None else data_format
        if mode == 'w':
            ndim = len(_axis_lengths)
            trace_properties, _axis_propdefs = get_trace_properties(ndim,
                properties, properties_add, properties_rm, axis_propdefs,
                similar_to)

        #self._is_open = True
        # TODO jsd._validate_js_dir after write complete
        # TODO load geometry from file properties xml

        compressor = get_trace_compressor(_axis_lengths[0], _data_format)
        jsd.properties = trace_properties
        jsd.axis_propdefs = _axis_propdefs
        jsd.compressor = compressor
        jsd.filename = filename
        jsd.mode = mode
        jsd.current_volume = -1
        jsd.header_length = get_header_length(jsd.properties)
        jsd.trace_length = get_trace_length(jsd.compressor)

        if mode == 'r' or mode == 'r+':
            filename = osp.join(jsd.path, JS_NAME_FILE)
            jsd.description = get_description(filename)

            jsd.mapped          = jsd._file_properties.is_mapped()
            jsd.data_type       = jsd._file_properties.data_type
            jsd.data_format     = _data_format
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
            filename = osp.join(jsd.path, JS_STATUS_FILE)
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

            return jsd

        if mode == 'w' and osp.isdir(filename):
            pass # TODO remote dataset for overwrite?

        if mode == 'w' and similar_to == "":
            jsd.mapped          = True if mapped is None else mapped
            jsd.data_type       = stock_dtype['CUSTOM'] if data_type is None else data_type
            jsd.data_format     = _data_format
            jsd.data_order      = "LITTLE_ENDIAN" if data_order is None else data_order
            jsd.axis_lengths    = axis_lengths
            jsd.axis_units      = axis_units
            jsd.axis_domains    = axis_domains
            jsd.axis_lstarts    = axis_lstarts
            jsd.axis_lincs      = axis_lincs
            jsd.axis_pstarts    = axis_pstarts
            jsd.axis_pincs      = axis_pincs
            jsd.data_properties = data_properties
            jsd.geom            = geometry
            jsd.secondaries     = ["."] if secondaries is None else secondaries

        elif mode == 'w' and similar_to != "":
            #jsdsim = JavaSeisDataset.open(similar_to)

            # special handling for data properties
            if len(data_properties) == 0:
                data_properties = jsdsim.data_properties
            else:
                assert len(data_properties_add) == 0
                assert len(data_properties_rm) == 0
            # TODO need test here
            if len(data_properties_add) != 0:
                assert len(data_properties) == 0
                for prop in data_properties_add:
                    if prop not in data_properties:
                        data_properties.append(prop)

            jsd.mapped          = jsdsim.mapped if mapped is None else mapped
            jsd.data_type       = jsdsim.data_type if data_type is None else data_type
            jsd.data_format     = _data_format
            jsd.data_order      = jsdsim.data_order if data_order is None else data_order
            jsd.axis_lengths    = axis_lengths
            jsd.axis_units      = jsdsim.axis_units if len(axis_units) == 0 else axis_units
            jsd.axis_domains    = jsdsim.axis_domains if len(axis_domains) == 0 else axis_domains
            jsd.axis_lstarts    = jsdsim.axis_lstarts if len(axis_lstarts) == 0 else axis_lstarts
            jsd.axis_lincs      = jsdsim.axis_lincs if len(axis_lincs) == 0 else axis_lincs
            jsd.axis_pstarts    = jsdsim.axis_pstarts if len(axis_pstarts) == 0 else axis_pstarts
            jsd.axis_pincs      = jsdsim.axis_pincs if len(axis_pincs) == 0 else axis_pincs
            jsd.data_properties = data_properties
            jsd.geom            = jsdsim.geom if geometry is None else geometry
            jsd.secondaries     = jsdsim.secondaries if secondaries is None else secondaries
            nextents            = jsdsim.trc_extents if nextents == 0 else nextents

        if mode == 'w':
            ndim = len(jsd.axis_lengths)
            assert ndim >= 3
            assert len(jsd.axis_propdefs) == ndim or len(jsd.axis_propdefs) == 0
            assert len(jsd.axis_units)    == ndim or len(jsd.axis_units)    == 0
            assert len(jsd.axis_domains)  == ndim or len(jsd.axis_domains)  == 0
            assert len(jsd.axis_lstarts)  == ndim or len(jsd.axis_lstarts)  == 0
            assert len(jsd.axis_lincs)    == ndim or len(jsd.axis_lincs)    == 0
            assert len(jsd.axis_pstarts)  == ndim or len(jsd.axis_pstarts)  == 0
            assert len(jsd.axis_pincs)    == ndim or len(jsd.axis_pincs)    == 0

            hassim = False if similar_to == "" else True
            JavaSeisDataset.write(jsd, nextents, ndim, description, properties, hassim)
            return jsd

    @staticmethod
    def write(jsd, nextents, ndim, description, properties, hassim):
        # initialize axes if not set yet
        if len(jsd.axis_units) == 0:
            jsd.axis_units = [stock_unit["UNKNOWN"]] * ndim
        if len(jsd.axis_domains) == 0:
            jsd.axis_domains = [stock_domain["UNKNOWN"]] * ndim
        if len(jsd.axis_lstarts) == 0:
            jsd.axis_lstarts = np.ones(ndim, dtype='int64')
        if len(jsd.axis_lincs) == 0:
            jsd.axis_lincs = np.ones(ndim, dtype='int64')
        if len(jsd.axis_pstarts) == 0:
            jsd.axis_pstarts = np.zeros(ndim, dtype='float64')
        if len(jsd.axis_pincs) == 0:
            jsd.axis_pincs = np.ones(ndim, dtype='float64')

        # description, if not set by user, we grab it from the filename
        if len(description) == 0:
            fn = jsd.filename[:-3] if jsd.filename.endswith(".js") else jsd.filename
            fn = fn.split('/')[-1] # TODO how about windows \
            jsd.description = fn.split('@')[-1]
        else:
            jsd.description = description

        # data is initialized to empty
        jsd.has_traces = False

        # secondaries, if not set by user, we use primary storage ["."]
        if jsd.secondaries is None:
            jsd.secondaries = ["."]
        if len(jsd.secondaries) < 1:
            raise ValueError("secondaries list length < 1")

        # choose default number of exents (heuristic)
        nextents = get_nextents(jsd.axis_lengths, jsd.data_format) if nextents == 0 else nextents
        nextents = min(nextents, np.prod(jsd.axis_lengths[2:]))

        # trace and header extents
        jsd.trc_extents = make_extents(nextents, jsd.secondaries,
            jsd.filename, jsd.axis_lengths, jsd.trace_length, "TraceFile")
        jsd.hdr_extents = make_extents(nextents, jsd.secondaries,
            jsd.filename, jsd.axis_lengths, jsd.header_length, "TraceHeaders")

        # trace map
        jsd.map = np.zeros(jsd.axis_lengths[2], dtype='int32')

        # create the various xml files and directories
        make_primary_dir(jsd)
        make_extent_dirs(jsd)
        create_map(jsd)
        write_file_properties(jsd)
        write_name_properties(jsd)
        write_status_properties(jsd)
#        write_extent_manager(jsd)
        write_virtual_folders(jsd)

    @staticmethod
    def _validate_js_dir(path):
        """Gets called during the construction of this object instance"""
        def js_missing(f):
            raise IOError("Missing: {}".format(f))
        files = os.listdir(path)

        if JS_FILE_PROPERTIES_XML not in files:
            js_missing(JS_FILE_PROPERTIES_XML)

        if JS_HISTORY_XML not in files:
            js_missing(JS_HISTORY_XML)

        if JS_TRACE_DATA_XML not in files:
            js_missing(JS_TRACE_DATA_XML)

        if JS_TRACE_HEADERS_XML not in files:
            js_missing(JS_TRACE_HEADERS_XML)

        if JS_NAME_FILE not in files:
            js_missing(JS_NAME_FILE)

        if JS_STATUS_FILE not in files:
            js_missing(JS_STATUS_FILE)

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
        trcfmt = self._js_dataset.data_format
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
                    extents[i] = extent_dict(name, path, i, start, size)

    # TODO add missing extents (i.e. extents with all empty frames)

    # the last extent might be a different size
    extent = extents[nextents-1]
    extent['size'] = maxpos - extent['start']

    return extents


def extent_dict(name, path, index, start, size):
    return {'name': name, 'path': path, 'index': index, 'start': start,
            'size': size}


def make_extents(nextents, secondaries, filename, axis_lengths,
    bytes_per_trace, basename):
    isec, nsec = 0, len(secondaries) - 1
    total_size = np.prod(axis_lengths[1:]) * bytes_per_trace
    frames_per_extent = math.ceil(np.prod(axis_lengths[2:]) / nextents)
    extent_size = frames_per_extent * axis_lengths[1] * bytes_per_trace
    extents = []
    for i in range(nextents):
        name = "basename{}".format(i)
        path = osp.join(extent_dir(secondaries[isec], filename), name)
        index = i
        start = index * extent_size
        size = min(extent_size, total_size)
        extents.append(extent_dict(name, path, index, start, size))
        isec = 0 if isec == nsec else isec + 1
        total_size -= extent_size
    return extents


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


def get_header_length(trace_headers):
    """
    -i- trace_headers : dict, element is TraceHeader object
    """
    total_bytes = 0
    for key, th in trace_headers.items():
        total_bytes += th.format_size
    return total_bytes


def get_nextents(dims, fmt):
    n = np.prod(dims) * FORMAT_BYTE[fmt] / (2.0 * 1024.0**3) + 10.0
    return math.ceil(np.clip(n, 1, 256))


def make_directory(abspath, force=False):
    if osp.isdir(abspath):
        if force: # delete old and create new
            shutil.rmtree(abspath)
            os.mkdir(abspath)
        else:
            warnings.warn("Directory already exists: {}".format(abspath))
    else:
        os.mkdir(abspath)


def make_primary_dir(jsd):
    make_directory(jsd.filename)


def make_extent_dirs(jsd):
    for path in jsd.secondaries:
        extpath = extent_dir(path, jsd.filename)
        make_directory(extpath)


def create_map(jsd):
    fn = osp.join(jsd.filename, JS_TRACE_MAP)
    nframes = np.prod(jsd.axis_lengths[2:])
    array = np.zeros(nframes, dtype='int32')
    with open(fn, 'wb') as f:
        array.tofile(f) # TODO need test by read


def write_file_properties(jsd):
    root = etree.Element("parset", name="JavaSeis Metadata")
    fps = etree.SubElement(root, "parset", name=JS_FILE_PROPERTIES)

    # translate ProMax property labels to JavaSeis axis labels
    axis_labels = []
    for key in jsd.axis_propdefs:
        label = dictPMtoJS[key] if key in dictPMtoJS else key
        axis_labels.append(label)

    add_child_par(fps, "Comments",          "string", " \"JavaSeis.py - JavaSeis File Propertties 2006.3\" ")
    add_child_par(fps, "JavaSeisVersion",   "string", " 2006.3 ")
    add_child_par(fps, "DataType",          "string",  " {} ".format(jsd.data_type))
    add_child_par(fps, "TraceFormat",       "string",  " {} ".format(DATA_FORMAT_TO_TRACE_FORMAT[jsd.data_format]))
    add_child_par(fps, "ByteOrder",         "string",  " {} ".format(jsd.data_order))
    add_child_par(fps, "Mapped",            "boolean", " {} ".format(str(jsd.mapped).lower()))
    add_child_par(fps, "DataDimensions",    "int",     " {} ".format(len(jsd.axis_lengths)))
    add_child_par(fps, "AxisLabels",        "string",  format_axes(axis_labels))
    add_child_par(fps, "AxisUnits",         "string",  format_axes(jsd.axis_units))
    add_child_par(fps, "AxisDomains",       "string",  format_axes(jsd.axis_domains))
    add_child_par(fps, "AxisLengths",       "long",    format_axes(jsd.axis_lengths))
    add_child_par(fps, "LogicalOrigins",    "long",    format_axes(jsd.axis_lstarts))
    add_child_par(fps, "LogicalDeltas",     "long",    format_axes(jsd.axis_lincs))
    add_child_par(fps, "PhysicalOrigins",   "double",  format_axes(jsd.axis_pstarts))
    add_child_par(fps, "PhysicalDeltas",    "double",  format_axes(jsd.axis_pincs))
    add_child_par(fps, "HeaderLengthBytes", "int",     " {} ".format(jsd.header_length))

    # trace properties
    tps = etree.SubElement(root, "parset", name="TraceProperties")
    i = 0
    for key, th in jsd.properties.items():
        add_child_trace_property(tps, i, th)
        i += 1

    # custom properties
    cps = etree.SubElement(root, "parset", name="CustomProperties")
    for prop in jsd.data_properties:
        add_child_par(cps, prop.label, prop.format, " {} ".format(prop.value))
        # TODO need test if SeisSpace can load the Stacked = "True"

    # 3-point geometry
    if jsd.geom is not None:
        geometry = etree.SubElement(cps, "parset", name="Geometry")
        add_child_par(geometry, "u1", "long",   " {} ".format(jsd.geom.u1))
        add_child_par(geometry, "un", "long",   " {} ".format(jsd.geom.un))
        add_child_par(geometry, "v1", "long",   " {} ".format(jsd.geom.v1))
        add_child_par(geometry, "vn", "long",   " {} ".format(jsd.geom.vn))
        add_child_par(geometry, "w1", "long",   " {} ".format(jsd.geom.w1))
        add_child_par(geometry, "wn", "long",   " {} ".format(jsd.geom.wn))
        add_child_par(geometry, "ox", "double", " {} ".format(jsd.geom.ox))
        add_child_par(geometry, "oy", "double", " {} ".format(jsd.geom.oy))
        add_child_par(geometry, "oz", "double", " {} ".format(jsd.geom.oz))
        add_child_par(geometry, "ux", "double", " {} ".format(jsd.geom.ux))
        add_child_par(geometry, "uy", "double", " {} ".format(jsd.geom.uy))
        add_child_par(geometry, "uz", "double", " {} ".format(jsd.geom.uz))
        add_child_par(geometry, "vx", "double", " {} ".format(jsd.geom.vx))
        add_child_par(geometry, "vy", "double", " {} ".format(jsd.geom.vy))
        add_child_par(geometry, "vz", "double", " {} ".format(jsd.geom.vz))
        add_child_par(geometry, "wx", "double", " {} ".format(jsd.geom.wx))
        add_child_par(geometry, "wy", "double", " {} ".format(jsd.geom.wy))
        add_child_par(geometry, "wz", "double", " {} ".format(jsd.geom.wz))

    fn = osp.join(jsd.filename, JS_FILE_PROPERTIES_XML)
    write_etree_to_file(fn, root)


def add_child_trace_property(parent, index, trace_header):
    """
    -i- parent : lxml.etree.Element or SubElement
    -i- index : int
    -i- trace_header : object, TraceHeader
    """
    header = etree.SubElement(parent, "parset", name="entry_{}".format(index))
    add_child_par(header, "label",        "string", " {} ".format(trace_header.label))
    add_child_par(header, "description",  "string", " \"{}\" ".format(trace_header.description))
    add_child_par(header, "format",       "string", " {} ".format(trace_header.format))
    add_child_par(header, "elementCount", "int", " {} ".format(trace_header.element_count))
    add_child_par(header, "byteOffset",   "int", " {} ".format(trace_header.byte_offset))


def format_axes(axes):
    """
    -i- axes : list, e.g. ["SAMPLE", "TRACE", "FRAME"], [64,32, 16]
    """
    prefix = "\n" + " " * 6
    append = "\n" + " " * 4
    labels = ""
    for axis in axes:
        labels += prefix + str(axis)
    labels += append
    return labels


def write_virtual_folders(jsd):
    root = etree.Element("parset", name=JS_VIRTUAL_FOLDERS)
    add_child_par(root, "NDIR", "int", " {} ".format(len(jsd.secondaries)))
    for i in range(len(jsd.secondaries)):
        value = " {},READ_WRITE ".format(jsd.secondaries[i])
        add_child_par(root, "FILESYSTEM-{}".format(i), "string", value)
    add_child_par(root, "Version",   "string", " 2006.2 ")
    add_child_par(root, "Header",    "string", " \"VFIO org.javaseis.VirtualFolder 2006.2\" ")
    add_child_par(root, "Type",      "string", " SS ")
    add_child_par(root, "POLICY_ID", "string", " RANDOM ")
    nb = jsd.axis_lengths[0] * FORMAT_BYTE[jsd.data_format] + jsd.header_length
    nb *=  np.prod(jsd.axis_lengths[1:])
    add_child_par(root, "GLOBAL_REQUIRED_FREE_SPACE", "long", " {} ".format(nb))
    fn = osp.join(jsd.filename, JS_VIRTUAL_FOLDERS_XML)
    write_etree_to_file(fn, root)


def write_etree_to_file(fn, root):
    """
    -i- root : lxml.etree.Element
    """
    et = etree.ElementTree(root)
    et.write(fn, pretty_print=True)


def add_child_par(parent, name, fmt, value):
    """
    -i- parent : lxml.etree.Element or SubElement
    -i- name : string
    -i- fmt : string
    -i- value : string
    """
    child = etree.SubElement(parent, "par", name=name, type=fmt)
    #child.set("name", name)
    #child.set("type", fmt)
    child.text = value


def write_name_properties(jsd):
    fn = osp.join(jsd.filename, JS_NAME_FILE)
    print('name2 =', jsd.description)
    with open(fn, 'w') as f:
        f.writelines("#JavaSeis.py - JavaSeis File Properties 2006.3\n")
        f.writelines("#UTC {}\n".format(datetime.now(pytz.utc)))
        f.writelines("DescriptiveName={}\n".format(jsd.description))


def write_status_properties(jsd):
    fn = osp.join(jsd.filename, JS_STATUS_FILE)
    with open(fn, 'w') as f:
        f.writelines("#JavaSeis.py - JavaSeis File Properties 2006.3\n")
        f.writelines("#UTC {}\n".format(datetime.now(pytz.utc)))
        f.writelines("HasTraces={}\n".format(jsd.has_traces))


if __name__ == '__main__':
    testpath = "/home/asbjorn/datasets/2hots.js"
    if not osp.exists(testpath):
        print("'{0}' dataset does not exists..".format(testpath))
    jsDataset = JavaSeisDataset(testpath)
    print(jsDataset)
