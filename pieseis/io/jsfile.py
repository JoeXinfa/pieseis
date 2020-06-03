import os
import os.path as osp
import struct
import warnings
import math
import shutil
import pytz
from datetime import datetime
import collections
from collections import OrderedDict as odict

# TODO python collections.namedtuple check it out and
# compare with julia NamedTuple as used in TeaSeis.jl

from lxml import etree
import numpy as np

from .properties import (FileProperties, TraceProperties, CustomProperties,
    VirtualFolders, TraceFileXML, TraceHeadersXML, TraceHeader)
from .trace_compressor import (get_trace_compressor, get_trace_length,
    unpack_frame, pack_frame)
from .compat import dictJStoPM, dictPMtoJS
from .stock_props import (minimal_props, stock_props, stock_dtype,
    stock_unit, stock_domain)
from ..utils.cartesian import cartesian

DOT_XML = ".xml"
JS_FILE_PROPERTIES = "FileProperties"
JS_TRACE_DATA = "TraceFile"
JS_TRACE_HEADERS = "TraceHeaders"
JS_VIRTUAL_FOLDERS = "VirtualFolders"
JS_HISTORY = "History"
JS_COMMENT = "PieSeis.py - JavaSeis File Propertties 2006.3"

# constant filenames
JS_TRACE_MAP = "TraceMap"
JS_NAME_FILE = "Name.properties"
JS_STATUS_FILE = "Status.properties"
JS_FILE_PROPERTIES_XML = JS_FILE_PROPERTIES + DOT_XML
JS_TRACE_DATA_XML = JS_TRACE_DATA + DOT_XML
JS_TRACE_HEADERS_XML = JS_TRACE_HEADERS + DOT_XML
JS_VIRTUAL_FOLDERS_XML = JS_VIRTUAL_FOLDERS + DOT_XML
JS_HISTORY_XML = JS_HISTORY + DOT_XML

JS_DATA_FORMAT = "int16"

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

# https://docs.python.org/3/library/struct.html
DATA_ORDER_TO_CHAR = {
    "LITTLE_ENDIAN" : "<",
    "BIG_ENDIAN"    : ">",
    "NATIVE"        : "=",
    "NETWORK"       : "!"
}


class JavaSeisDataset(object):
    """
    Class to host a JavaSeis dataset. This class will be used by both the
    Reader and Writer classes.
    """
    def __init__(self, filename):
        self.path = filename

    @staticmethod
    def _check_io(filename, mode):
        if not filename.endswith(".js"):
            warnings.warn("JavaSeis filename does not end with '.js'")
        if mode == 'r' or mode == 'r+':
            if not osp.isdir(filename):
                raise IOError("JavaSeis filename is not directory")
            if not os.access(filename, os.R_OK):
                raise IOError("Missing read access")
        elif mode == 'w':
            #if osp.exists(filename):
            #    raise IOError("Path for JavaSeis dataset already exists")
            parent_directory = osp.dirname(filename)
            if not os.access(parent_directory, os.W_OK):
                raise IOError("Missing write access in {}".format(parent_directory))
        else:
            raise ValueError("unrecognized mode: {}".format(mode))

    @classmethod
    def open(cls, filename,
        mode                = "r",
        description         = "",
        is_mapped           = None,
        data_type           = None,
        data_format         = None,
        data_order          = None,
        axis_lengths        = [],
        axis_propdefs       = odict(),
        axis_units          = [],
        axis_domains        = [],
        axis_lstarts        = [],
        axis_lincs          = [],
        axis_pstarts        = [],
        axis_pincs          = [],
        data_properties     = [],
        properties          = odict(),
        geometry            = None,
        secondaries         = None,
        nextents            = 0,
        similar_to          = "",
        properties_add      = odict(),
        properties_rm       = odict(),
        data_properties_add = [],
        data_properties_rm  = []):
        """
        -i- filename : string, path of the JavaSeis dataset e.g. "data.js"
        -i- mode : string, "r" read, "w" write/create, "r+" read and write
        """
        JavaSeisDataset._check_io(filename, mode)
        jsd = JavaSeisDataset(filename)
        if mode == 'r' or mode == 'r+':
            jsd.is_valid = jsd._validate_js_dir(jsd.path)
            jsd._read_properties()
            trace_properties = jsd._trace_properties._trace_headers
            axis_labels = jsd._file_properties.axis_labels
            _axis_propdefs = JavaSeisDataset.get_axis_propdefs(trace_properties, axis_labels)
            _axis_lengths = jsd._file_properties.axis_lengths
            _data_format = TRACE_FORMAT_TO_DATA_FORMAT[jsd._file_properties.trace_format]
        elif mode == 'w' and similar_to == "":
            _axis_lengths = axis_lengths
            _data_format = JS_DATA_FORMAT if data_format is None else data_format
        elif mode == 'w' and similar_to != "":
            jsdsim = JavaSeisDataset.open(similar_to)
            _axis_lengths = jsdsim.axis_lengths if len(axis_lengths) == 0 else axis_lengths
            _data_format = jsdsim.data_format if data_format is None else data_format
        if mode == 'w':
            ndim = len(_axis_lengths)
            trace_properties, _axis_propdefs = JavaSeisDataset.get_trace_properties(
            ndim, properties, properties_add, properties_rm, axis_propdefs, similar_to)

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
        jsd.header_length = jsd.get_header_length(jsd.properties)
        jsd.trace_length = get_trace_length(jsd.compressor)

        if mode == 'r' or mode == 'r+':
            filename = osp.join(jsd.path, JS_NAME_FILE)
            jsd.description = JavaSeisDataset.get_description(filename)

            jsd.is_mapped       = jsd._file_properties.is_mapped
            jsd.data_type       = jsd._file_properties.data_type
            jsd.data_format     = _data_format
            jsd.data_order      = jsd._file_properties.byte_order
            jsd.data_order_char = DATA_ORDER_TO_CHAR[jsd.data_order]
            jsd.axis_lengths    = jsd._file_properties.axis_lengths
            jsd.axis_units      = jsd._file_properties.axis_units
            jsd.axis_domains    = jsd._file_properties.axis_domains
            jsd.axis_lstarts    = jsd._file_properties.logical_origins
            jsd.axis_lincs      = jsd._file_properties.logical_deltas
            jsd.axis_pstarts    = jsd._file_properties.physical_origins
            jsd.axis_pincs      = jsd._file_properties.physical_deltas
            jsd.data_properties = jsd._custom_properties.data_properties
            jsd.geom            = None
            jsd.is_regular      = True

            jsd.has_traces = False
            filename = osp.join(jsd.path, JS_STATUS_FILE)
            if osp.isfile(filename):
                jsd.has_traces = JavaSeisDataset.get_status(filename)

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
            jsd.is_mapped       = True if is_mapped is None else is_mapped
            jsd.data_type       = stock_dtype['CUSTOM'] if data_type is None else data_type
            jsd.data_format     = _data_format
            jsd.data_order      = "LITTLE_ENDIAN" if data_order is None else data_order
            jsd.data_order_char = DATA_ORDER_TO_CHAR[jsd.data_order]
            jsd.axis_lengths    = _axis_lengths
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

            jsd.is_mapped       = jsdsim.is_mapped if is_mapped is None else is_mapped
            jsd.data_type       = jsdsim.data_type if data_type is None else data_type
            jsd.data_format     = _data_format
            jsd.data_order      = jsdsim.data_order if data_order is None else data_order
            jsd.data_order_char = DATA_ORDER_TO_CHAR[jsd.data_order]
            jsd.axis_lengths    = _axis_lengths
            jsd.axis_units      = jsdsim.axis_units if len(axis_units) == 0 else axis_units
            jsd.axis_domains    = jsdsim.axis_domains if len(axis_domains) == 0 else axis_domains
            jsd.axis_lstarts    = jsdsim.axis_lstarts if len(axis_lstarts) == 0 else axis_lstarts
            jsd.axis_lincs      = jsdsim.axis_lincs if len(axis_lincs) == 0 else axis_lincs
            jsd.axis_pstarts    = jsdsim.axis_pstarts if len(axis_pstarts) == 0 else axis_pstarts
            jsd.axis_pincs      = jsdsim.axis_pincs if len(axis_pincs) == 0 else axis_pincs
            jsd.data_properties = data_properties
            jsd.geom            = jsdsim.geom if geometry is None else geometry
            jsd.secondaries     = jsdsim.secondaries if secondaries is None else secondaries
            nextents            = len(jsdsim.trc_extents) if nextents == 0 else nextents

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
        # TODO check properties is not used...?
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
        nextents = jsd.get_nextents(jsd.axis_lengths, jsd.data_format) if nextents == 0 else nextents
        nextents = min(nextents, np.prod(jsd.axis_lengths[2:]))

        # trace and header extents
        jsd.trc_extents = JavaSeisDataset.make_extents(nextents, jsd.secondaries,
            jsd.filename, jsd.axis_lengths, jsd.trace_length, "TraceFile")
        jsd.hdr_extents = JavaSeisDataset.make_extents(nextents, jsd.secondaries,
            jsd.filename, jsd.axis_lengths, jsd.header_length, "TraceHeaders")

        # trace map
        jsd.map = np.zeros(jsd.axis_lengths[2], dtype='int32')

        # create the various xml files and directories
        jsd.make_primary_dir()
        jsd.make_extent_dirs()
        jsd.create_map()
        jsd.write_file_properties()
        jsd.write_name_properties()
        jsd.write_status_properties()
        jsd.write_extent_manager()
        jsd.write_virtual_folders()

    @staticmethod
    def get_nextents(dims, fmt):
        n = np.prod(dims) * FORMAT_BYTE[fmt] / (2.0 * 1024.0**3) + 10.0
        return math.ceil(np.clip(n, 1, 256))

    @staticmethod
    def get_header_length(trace_headers):
        """
        -i- trace_headers : dict, element is TraceHeader object
        """
        total_bytes = 0
        for key, th in trace_headers.items():
            total_bytes += th.format_size
        return total_bytes

    @property
    def nframes(self):
        """
        Return the number of frames, which is useful for iterating over
        all frames in a JavaSeis dataset.
        """
        return np.prod(self.axis_lengths[2:])

    @property
    def ndim(self):
        """
        Return the dimension
        """
        return len(self.axis_lengths)

    def ind2sub(self, index):
        """
        -i- index : integer, absolute index of frame, range [1, nframes]
        -o- sub : tuple, (ifrm, ivol, ihyp)
        This is useful for looping over all frames in 4+ dimensions.
        """
        axis_arrays = []
        for i in range(2, self.ndim):
            start = self.axis_lstarts[i]
            step = self.axis_lincs[i]
            num = self.axis_lengths[i]
            stop = start + step * (num - 1)
            array = np.linspace(start, stop, num)
            axis_arrays.append(array)
        cart = cartesian(*axis_arrays)
        cart = cart.astype(int) # is this good?
        assert 1 <= index <= self.nframes
        return tuple(cart[index-1])

    def sub2ind(self, indices):
        """
        -i- indices : tuple, (ifrm, ivol, ihyp)
        -o- index : integer, absolute index of frame, range [1, nframes]
        """
        n = len(indices)
        assert n == self.ndim - 2
        for i in range(n):
            start = self.axis_lstarts[2+i]
            step = self.axis_lincs[2+i]
            num = self.axis_lengths[2+i]
            stop = start + step * (num - 1)
            assert start <= indices[i] <= stop
            
        a = indices[0] - self.axis_lstarts[2]
        b = self.axis_lincs[2]
        idx_lin, idx_mod = a // b, a % b
        assert idx_mod == 0
        idx_lin += 1
        
        for i in range(1,n):
            a = indices[i] - self.axis_lstarts[2+i]
            b = self.axis_lincs[2+i]
            idx_lin_i, idx_mod = a // b, a % b
            assert idx_mod == 0
            idx_lin += idx_lin_i * np.prod(self.axis_lengths[2:2+i])
        return idx_lin

    @staticmethod
    def _validate_js_dir(path):
        """Gets called during the construction of this object instance"""
        def js_missing(f, warn=False):
            if warn:
                warnings.warn("Missing: {}".format(f))
            else:
                raise IOError("Missing: {}".format(f))
        files = os.listdir(path)

        if JS_FILE_PROPERTIES_XML not in files:
            js_missing(JS_FILE_PROPERTIES_XML)

        if JS_HISTORY_XML not in files:
            js_missing(JS_HISTORY_XML, warn=True)

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

    def read_frame_trcs(self, iframe):
        """
        -i- iframe : int, index of frame, range is [1, nframe]
        -o- trcs : array, float32, shape (ntrace, nsample)
        """
        assert self.mode != 'w'
        fold = self._fold(iframe)
        if fold == 0:
            return None
        frame_size = self.trace_length * self.axis_lengths[1]
        offset = (iframe - 1) * frame_size
        extent = get_extent_index(self.trc_extents, offset)
        offset -= extent['start']
        if self.data_format == "int16":
            with open(extent['path'], "rb") as f:
                array = unpack_frame(f, offset, self.compressor, fold)
            return array
        elif self.data_format == "float32":
            pass # TODO
        else:
            raise ValueError("Unsupported trace format".format(self.data_format))

    def read_frame_hdrs(self, iframe):
        assert self.mode != 'w'
        fold = self._fold(iframe)
        if fold == 0:
            return 0
        #TODO set trace type dead from fold+1 to frame length

        frame_size = self.header_length * self.axis_lengths[1]
        offset = (iframe - 1) * frame_size
        extent = get_extent_index(self.hdr_extents, offset)
        offset -= extent['start']
        with open(extent['path'], "rb") as f:
            f.seek(offset)
            frame_bytes = f.read(self.header_length * fold)
        #return self.unpack_frame_hdrs(frame_bytes, fold)
        #return frame_bytes # bytes, immutable, read-only
        return bytearray(frame_bytes) # mutable, read/write

    def _read_frame(self, iframe):
        trcs = self.read_frame_trcs(iframe)
        hdrs = self.read_frame_hdrs(iframe)
        return trcs, hdrs

    def read_frame(self, indices):
        """
        -i- indices : tuple, indices of the frame to read
            For 3D dataset, (frm_idx)
            For 4D dataset, (frm_idx, vol_idx)
            For 5D dataset, (frm_idx, vol_idx, hyp_idx)
        """
        iframe = self.sub2ind(indices)
        return self._read_frame(iframe)

    def unpack_frame_hdrs(self, frame_bytes, fold):
        """
        -o- frame_headers : list, element is namedtuple of headers, one for one trace
        Is there any use case of this output?
        """
        TraceHeaders = collections.namedtuple("TraceHeaders", self.properties.keys())
        frame_headers = []
        for itrace in range(fold):
            trace_headers = []
            for key in TraceHeaders._fields:
                th = self.properties[key]
                fmt = self.data_order_char + th._format_char
                offset = self.header_length * itrace + th._byte_offset
                header_value = struct.unpack_from(fmt, frame_bytes, offset=offset)[0]
                trace_headers.append(header_value)
            nt = TraceHeaders(*trace_headers)
            frame_headers.append(nt)
        return frame_headers

    def _get_trace_header(self, header_label, itrace, frame_bytes):
        assert header_label in self.properties

#        b1 = self.header_length * (itrace - 1)
#        b2 = b1 + self.header_length
#        trace_bytes = frame_bytes[b1:b2]
#        th = self.properties[header_label]
#        b1 = th._byte_offset # is it faster than th.byte_offset? safe?
#        b2 = b1 + th._format_size
#        header_bytes = trace_bytes[b1:b2]
#        fmt = self.data_order_char + th._format_char
#        return struct.unpack(fmt, header_bytes)[0] # tuple first value

        th = self.properties[header_label]
        fmt = self.data_order_char + th._format_char
        offset = self.header_length * (itrace - 1) + th._byte_offset
        return struct.unpack_from(fmt, frame_bytes, offset=offset)

    def get_trace_header(self, header_label, itrace, frame):
        if type(frame) == int:
            iframe = frame
            frame_bytes = self.read_frame_hdrs(iframe)
        elif type(frame) == tuple:
            iframe = self.sub2ind(frame)
            frame_bytes = self.read_frame_hdrs(iframe)
        elif type(frame) == bytearray:
            frame_bytes = frame
        else:
            raise TypeError
        return self._get_trace_header(header_label, itrace, frame_bytes)

    def set_trace_header(self, header_value, header_label, itrace, iframe):
        assert header_label in self.properties
        b1 = self.header_length * (itrace - 1)
        th = self.properties[header_label]
        b1 += th._byte_offset # offset within this frame

        fmt = self.data_order_char + th._format_char
        header_value = th.cast_value(header_value)
        header_bytes = struct.pack(fmt, header_value)

        # write file
        frame_size = self.header_length * self.axis_lengths[1]
        offset = (iframe - 1) * frame_size
        extent = get_extent_index(self.hdr_extents, offset)
        offset -= extent['start']
        filename = extent['path']
        mode = "r+b" if osp.isfile(filename) else "wb"
        with open(filename, mode) as f:
            f.seek(offset + b1)
            f.write(header_bytes)

    def set_header_in_frame(self, header, val, itrc, hin, hou=None):
        if hou is None:
            # in-place modify an existing header
            self._set_header_in_frame(header, val, itrc, hin)
        else:
            # add a new header by append
            self.add_header_to_frame(header, val, itrc, hin, hou)

    def _set_header_in_frame(self, header, val, itrc, hin):
        """
        -i- header : TraceHeader/string
        -i- val : int/float/bytes, header value to write
        -i- itrc : integer, the i-th trace in frame, range [1, fold]
        -i- hin : bytearray, headers bytes of the frame, in-place set
        """
        # TODO handle header.element_count > 1 ?
        header, val = self._set_header_prep(header, val)
        b1 = self.header_length * (itrc - 1)
        b1 += header._byte_offset # offset within this frame
        b2 = b1 + header._format_size
        hin[b1:b2] = val

    def add_header_to_frame(self, header, val, itrc, hin, hou):
        """
        -i- header : TraceHeader/string
        -i- val : int/float/bytes, header value to write
        -i- itrc : integer, the i-th trace in frame, range [1, fold]
        -i- hin : bytearray, headers bytes of the input frame
        -i- hou : bytearray, headers bytes of the output frame
        """
        header, val = self._set_header_prep(header, val)

        hlen_ou = self.header_length
        fold = len(hou) / hlen_ou
        hlen_in = int(len(hin) / fold)
        a1 = hlen_in * (itrc - 1)
        a2 = a1 + hlen_in
        b1 = hlen_ou * (itrc - 1)
        b2 = b1 + hlen_in
        b3 = b1 + hlen_ou

        hou[b1:b2] = hin[a1:a2]
        hou[b2:b3] = val

    def _set_header_prep(self, header, val):
        """
        -i- header : TraceHeader/string
        -i- val : int/float/bytes, header value to write
        -o- header : TraceHeader
        -o- val_bytes : bytes
        """
        # After this, header is TraceHeader
        if type(header) == str:
            header = self.properties[header]
        else:
            assert header.label in self.properties
        # Convert header value to bytes
        if type(val) == bytes:
            # This is to save time when the header value to set is the
            # same across frames, thus input is bytes already.
            val_bytes = val
        else:
            val = header.cast_value(val)
            fmt = self.data_order_char + header._format_char
            val_bytes = struct.pack(fmt, val)
        return header, val_bytes

    def write_frame(self, trcs, hdrs, fold, fidx):
        if type(fidx) == int:
            iframe = fidx
        elif type(fidx) == tuple:
            iframe = self.sub2ind(fidx)
        else:
            raise TypeError
        self._write_frame(trcs, hdrs, fold, iframe)

    def _write_frame(self, trcs, hdrs, fold, iframe):
        """
        -i- trcs : array, numpy 2D shape (ntrace, nsample)
        -i- hdrs : dict or bytearray
        -i- fold : integer, fold of this frame
        -i- iframe : integer, absolute index of frame
        """
        self.write_frame_trcs(trcs, fold, iframe)
        if type(hdrs) == dict:
            self.write_frame_hdrs(hdrs, fold, iframe)
        elif type(hdrs) == bytearray:
            self._write_frame_hdrs(hdrs, fold, iframe)
        else:
            raise TypeError
        self.save_map(iframe, fold) # tracemap or foldmap
        if not self.has_traces and fold > 0:
            self.has_traces = True
            self.write_status_properties()

    def write_frame_hdrs(self, hdrs, fold, iframe):
        """
        -i- hdrs : dict
        -i- fold : int
        -i- iframe : int
        """
        buffer = bytearray(self.header_length*fold)
        for i in range(fold):
            trace_offset = self.header_length * i
            for header, th in self.properties.items():
                value = 0.0
                if header in hdrs:
                    values = hdrs[header]
                    value = values[i] if isinstance(values, np.ndarray) else values
                value = th.cast_value(value)
                offset = trace_offset + th._byte_offset
                fmt = self.data_order_char + th._format_char
                struct.pack_into(fmt, buffer, offset, value)
        self._write_frame_hdrs(buffer, fold, iframe)

    def _write_frame_hdrs(self, buffer, fold, iframe):
        """
        -i- buffer : bytearray
        -i- fold : int
        -i- iframe : int
        """
        frame_size = self.header_length * self.axis_lengths[1]
        offset = (iframe - 1) * frame_size
        extent = get_extent_index(self.hdr_extents, offset)
        offset -= extent['start']
        filename = extent['path']
        mode = "r+b" if osp.isfile(filename) else "wb"
        with open(filename, mode) as f:
            f.seek(offset)
            f.write(buffer)

    def write_frame_trcs(self, trcs, fold, iframe):
        frame_size = self.trace_length * self.axis_lengths[1]
        offset = (iframe - 1) * frame_size
        extent = get_extent_index(self.trc_extents, offset)
        offset -= extent['start']
        filename = extent['path']
        mode = "r+b" if osp.isfile(filename) else "wb"
        if self.data_format == "int16":
            with open(filename, mode) as f:
                pack_frame(f, offset, self.compressor, fold, trcs)
        elif self.data_format == "float32":
            with open(filename, mode) as f:
                f.seek(offset)
                #f.write(trcs) # TODO convert
        else:
            raise ValueError("Unsupported trace format".format(self.data_format))

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
        self._trc_extents = JavaSeisDataset.get_extents(xml, secondaries, filename)

    def _set_hdr_extents(self):
        xml = self._trace_headers_xml
        secondaries = self._virtual_folders.secondary_folders
        filename = self.path
        self._hdr_extents = JavaSeisDataset.get_extents(xml, secondaries, filename)

    def _fold(self, iframe):
        if not self.is_mapped:
            return self.axis_lengths[1]
        self._read_map(iframe)
        index = self.get_map_position(iframe)
        return self.map[index]

    def fold(self, frame):
        """
        -i- frame : types
            If integer, absolute index of frame
            If tuple, indices of the frame, (ifrm, ivol, ihyp, ...)
            If bytearray, headers bytes of the frame
        """
        if type(frame) == int:
            iframe = frame
            fold = self._fold(iframe)
        elif type(frame) == tuple:
            iframe = self.sub2ind(frame)
            fold = self._fold(iframe)
        elif type(frame) == bytearray:
            fold = self._fold_from_hdrs(frame)
        else:
            raise TypeError
        return fold

    def _fold_from_hdrs(self, hdrs):
        return int(len(hdrs) / self.header_length) # safe?
        # Count the number of live traces
#        fold = 0
#        for i in range(n):
#            itrace = i + 1
#            trc_type = self._get_trace_header("TRC_TYPE", itrace, hdrs)
#            if trc_type == trace_type['live']:
#                fold += 1
#        return fold

    def _read_map(self, iframe):
        vol = self.get_volume_index(iframe)
        if vol == self.current_volume:
            return
        nframe = self.axis_lengths[2]
        position = (vol - 1) * nframe * 4
        fmt = "{}i".format(nframe)
        fn = osp.join(self.filename, JS_TRACE_MAP)
        with open(fn, 'rb') as f:
            f.seek(position)
            buffer = f.read(nframe * 4)
            self.map = np.array(struct.unpack(fmt, buffer), dtype='int32')
        self.current_volume = vol

    def _calc_total_ntrace_live(self):
        if self.is_mapped:
            total_ntrace_live = 0
            fmt = "{}i".format(self.nframes)
            fn = osp.join(self.filename, JS_TRACE_MAP)
            with open(fn, 'rb') as f:
                buffer = f.read(self.nframes * 4)
                folds = struct.unpack(fmt, buffer)
            for fold in folds:
                total_ntrace_live += fold
                if fold != self.axis_lengths[1]:
                    self.is_regular = False
        else:
            total_ntrace_live = self.total_ntrace
        return total_ntrace_live

    @property
    def total_ntrace(self):
        ntrace = self.axis_lengths[1]
        return ntrace * self.nframes

    def create_map(self):
        fn = osp.join(self.filename, JS_TRACE_MAP)
        array = np.zeros(self.nframes, dtype='int32')
        with open(fn, 'wb') as f:
            array.tofile(f)

    def save_map(self, iframe, fold):
        if self.is_mapped:
            if self.get_volume_index(iframe) == self.current_volume:
                self.map[self.get_map_position(iframe)] = np.int32(fold)
            position = (iframe - 1) * np.int32().itemsize
            fn = osp.join(self.filename, JS_TRACE_MAP)
            with open(fn, 'r+b') as f:
                f.seek(position)
                f.write(np.int32(fold)) # no need to pack?

    def get_volume_index(self, iframe):
        return int((iframe - 1) / self.axis_lengths[2]) + 1

    def get_map_position(self, iframe):
        return iframe - (self.get_volume_index(iframe) - 1) * self.axis_lengths[2] - 1

    def write_extent_manager(self):
        nb = np.prod(self.axis_lengths[1:]) * self.trace_length - 1
        root = etree.Element("parset", name="ExtentManager")
        add_child_par(root, "VFIO_VERSION", "string", " 2006.2 ")
        add_child_par(root, "VFIO_EXTSIZE", "long",   " {} ".format(self.trc_extents[0]['size']))
        add_child_par(root, "VFIO_MAXFILE", "int",    " {} ".format(len(self.trc_extents)))
        add_child_par(root, "VFIO_MAXPOS",  "long",   " {} ".format(nb))
        add_child_par(root, "VFIO_EXTNAME", "string", " {} ".format(JS_TRACE_DATA))
        add_child_par(root, "VFIO_POLICY",  "string", " RANDOM ")
        fn = osp.join(self.filename, JS_TRACE_DATA_XML)
        write_etree_to_file(fn, root)

        nb = np.prod(self.axis_lengths[1:]) * self.header_length - 1
        root = etree.Element("parset", name="ExtentManager")
        add_child_par(root, "VFIO_VERSION", "string", " 2006.2 ")
        add_child_par(root, "VFIO_EXTSIZE", "long",   " {} ".format(self.hdr_extents[0]['size']))
        add_child_par(root, "VFIO_MAXFILE", "int",    " {} ".format(len(self.hdr_extents)))
        add_child_par(root, "VFIO_MAXPOS",  "long",   " {} ".format(nb))
        add_child_par(root, "VFIO_EXTNAME", "string", " {} ".format(JS_TRACE_HEADERS))
        add_child_par(root, "VFIO_POLICY",  "string", " RANDOM ")
        fn = osp.join(self.filename, JS_TRACE_HEADERS_XML)
        write_etree_to_file(fn, root)

    def write_name_properties(self):
        fn = osp.join(self.filename, JS_NAME_FILE)
        with open(fn, 'w') as f:
            f.writelines("#{}\n".format(JS_COMMENT))
            f.writelines("#UTC {}\n".format(datetime.now(pytz.utc)))
            f.writelines("DescriptiveName={}\n".format(self.description))

    def write_status_properties(self):
        fn = osp.join(self.filename, JS_STATUS_FILE)
        with open(fn, 'w') as f:
            f.writelines("#{}\n".format(JS_COMMENT))
            f.writelines("#UTC {}\n".format(datetime.now(pytz.utc)))
            f.writelines("HasTraces={}\n".format(str(self.has_traces).lower()))

    def write_virtual_folders(self):
        root = etree.Element("parset", name=JS_VIRTUAL_FOLDERS)
        add_child_par(root, "NDIR", "int", " {} ".format(len(self.secondaries)))
        for i in range(len(self.secondaries)):
            value = " {},READ_WRITE ".format(self.secondaries[i])
            add_child_par(root, "FILESYSTEM-{}".format(i), "string", value)
        add_child_par(root, "Version",   "string", " 2006.2 ")
        add_child_par(root, "Header",    "string", " \"VFIO org.javaseis.VirtualFolder 2006.2\" ")
        add_child_par(root, "Type",      "string", " SS ")
        add_child_par(root, "POLICY_ID", "string", " RANDOM ")
        nb = self.axis_lengths[0] * FORMAT_BYTE[self.data_format] + self.header_length
        nb *=  np.prod(self.axis_lengths[1:])
        add_child_par(root, "GLOBAL_REQUIRED_FREE_SPACE", "long", " {} ".format(nb))
        fn = osp.join(self.filename, JS_VIRTUAL_FOLDERS_XML)
        write_etree_to_file(fn, root)

    def write_file_properties(self):
        root = etree.Element("parset", name="JavaSeis Metadata")
        fps = etree.SubElement(root, "parset", name=JS_FILE_PROPERTIES)

        # translate ProMax property labels to JavaSeis axis labels
        axis_labels = []
        for key in self.axis_propdefs:
            label = dictPMtoJS[key] if key in dictPMtoJS else key
            axis_labels.append(label)

        add_child_par(fps, "Comments",          "string", " \"{}\" ".format(JS_COMMENT))
        add_child_par(fps, "JavaSeisVersion",   "string", " 2006.3 ")
        add_child_par(fps, "DataType",          "string",  " {} ".format(self.data_type))
        add_child_par(fps, "TraceFormat",       "string",  " {} ".format(DATA_FORMAT_TO_TRACE_FORMAT[self.data_format]))
        add_child_par(fps, "ByteOrder",         "string",  " {} ".format(self.data_order))
        add_child_par(fps, "Mapped",            "boolean", " {} ".format(str(self.is_mapped).lower()))
        add_child_par(fps, "DataDimensions",    "int",     " {} ".format(len(self.axis_lengths)))
        add_child_par(fps, "AxisLabels",        "string",  format_axes(axis_labels))
        add_child_par(fps, "AxisUnits",         "string",  format_axes(self.axis_units))
        add_child_par(fps, "AxisDomains",       "string",  format_axes(self.axis_domains))
        add_child_par(fps, "AxisLengths",       "long",    format_axes(self.axis_lengths))
        add_child_par(fps, "LogicalOrigins",    "long",    format_axes(self.axis_lstarts))
        add_child_par(fps, "LogicalDeltas",     "long",    format_axes(self.axis_lincs))
        add_child_par(fps, "PhysicalOrigins",   "double",  format_axes(self.axis_pstarts))
        add_child_par(fps, "PhysicalDeltas",    "double",  format_axes(self.axis_pincs))
        add_child_par(fps, "HeaderLengthBytes", "int",     " {} ".format(self.header_length))

        # trace properties
        tps = etree.SubElement(root, "parset", name="TraceProperties")
        i = 0
        for key, th in self.properties.items():
            self.add_child_trace_property(tps, i, th)
            i += 1

        # custom properties
        cps = etree.SubElement(root, "parset", name="CustomProperties")
        for prop in self.data_properties:
            add_child_par(cps, prop.label, prop.format, " {} ".format(prop.value))
            # TODO need test if SeisSpace can load the Stacked = "True"

        # 3-point geometry
        if self.geom is not None:
            geometry = etree.SubElement(cps, "parset", name="Geometry")
            add_child_par(geometry, "u1", "long",   " {} ".format(self.geom.u1))
            add_child_par(geometry, "un", "long",   " {} ".format(self.geom.un))
            add_child_par(geometry, "v1", "long",   " {} ".format(self.geom.v1))
            add_child_par(geometry, "vn", "long",   " {} ".format(self.geom.vn))
            add_child_par(geometry, "w1", "long",   " {} ".format(self.geom.w1))
            add_child_par(geometry, "wn", "long",   " {} ".format(self.geom.wn))
            add_child_par(geometry, "ox", "double", " {} ".format(self.geom.ox))
            add_child_par(geometry, "oy", "double", " {} ".format(self.geom.oy))
            add_child_par(geometry, "oz", "double", " {} ".format(self.geom.oz))
            add_child_par(geometry, "ux", "double", " {} ".format(self.geom.ux))
            add_child_par(geometry, "uy", "double", " {} ".format(self.geom.uy))
            add_child_par(geometry, "uz", "double", " {} ".format(self.geom.uz))
            add_child_par(geometry, "vx", "double", " {} ".format(self.geom.vx))
            add_child_par(geometry, "vy", "double", " {} ".format(self.geom.vy))
            add_child_par(geometry, "vz", "double", " {} ".format(self.geom.vz))
            add_child_par(geometry, "wx", "double", " {} ".format(self.geom.wx))
            add_child_par(geometry, "wy", "double", " {} ".format(self.geom.wy))
            add_child_par(geometry, "wz", "double", " {} ".format(self.geom.wz))

        fn = osp.join(self.filename, JS_FILE_PROPERTIES_XML)
        write_etree_to_file(fn, root)

    @staticmethod
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

    def make_primary_dir(self):
        make_directory(self.filename, force=True)

    def make_extent_dirs(self):
        for path in self.secondaries:
            extpath = JavaSeisDataset.extent_dir(path, self.filename)
            #if extpath != self.filename: # C:Usersjoe vs C:/Users/joe
            if osp.normpath(extpath) != osp.normpath(self.filename):
                make_directory(extpath, force=True)

    @staticmethod
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
        properties = odict()
        
        # When provide similar_to file, the _property_defs is from that file.
        # The increment of byte_offset assumes the headers order in odict
        # (read from XML file) is consistent with the header byte_offset.
        # Luckily for the JS I see from SeisSpace so far, this is true.

        # trace properties, minimal set (as defined by SeisSpace / ProMAX)
        if similar_to == "":
            byte_offset = JavaSeisDataset.add_minimal_propset(properties, start_offset=0)
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

        # sort by label
        #properties = odict(sorted(properties.items()))
        # It seems SeisSpace JavaSeis requires trace header entries sorted
        # by their byte offset in the FileProperties.xml,
        # thus sort by label does not work.

        return properties, _axis_propdefs

    @staticmethod
    def add_minimal_propset(properties, start_offset=0):
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

    @staticmethod
    def get_description(filename):
        with open(filename, 'r') as f:
            for line in f:
                if line[0] != '#':
                    line = line.strip()
                    columns = line.split('=')
                    if columns[0] == "DescriptiveName":
                        return columns[1]
        return ""

    @staticmethod
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

    @staticmethod
    def get_axis_propdefs(trace_headers, axis_labels):
        axis_propdefs = odict() # need keep the order
        for i, label in enumerate(axis_labels):
            axis_propdefs[label] = JavaSeisDataset.get_axis_propdef(trace_headers, label, i+1)
        return axis_propdefs

    @staticmethod
    def get_axis_propdef(trace_headers, label, dim):
        # map from JavaSeis axis name to ProMax property label
        plabel = dictJStoPM[label] if label in dictJStoPM else label

        for key, value in trace_headers.items():
            if value.label == plabel:
                return value

        # The sample and trace labels do not need a corresponding trace property.
        # Therefore, these should be considered valid datasets.
        if dim == 1 or dim == 2:
            return TraceHeader(values=(label, label, 'INTEGER', 1, 0))

        raise ValueError("Malformed JavaSeis: axis props, axis label={} has no "
                         "corresponding trace property".format(label))

    @staticmethod
    def make_extents(nextents, secondaries, filename, axis_lengths,
        bytes_per_trace, basename):
        isec, nsec = 0, len(secondaries) - 1
        total_size = np.prod(axis_lengths[1:]) * bytes_per_trace
        frames_per_extent = math.ceil(np.prod(axis_lengths[2:]) / nextents)
        extent_size = frames_per_extent * axis_lengths[1] * bytes_per_trace
        extents = []
        for i in range(nextents):
            name = "{}{}".format(basename, i)
            path = osp.join(JavaSeisDataset.extent_dir(secondaries[isec], filename), name)
            index = i
            start = index * extent_size
            size = min(extent_size, total_size)
            extents.append(extent_dict(name, path, index, start, size))
            isec = 0 if isec == nsec else isec + 1
            total_size -= extent_size
        return extents

    @staticmethod
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
        extents = [extent_dict()] * nextents
        for secondary in secondaries:
            base_extpath = JavaSeisDataset.extent_dir(secondary, filename)
            if osp.isdir(base_extpath):
                names = get_names(base_extpath, basename)
                for name in names:
                    i = int(name[len(basename):])
                    if i < nextents:
                        start = i * size
                        path = osp.join(base_extpath, name)
                        extents[i] = extent_dict(name, path, i, start, size)

        # Add missing extents (i.e. extents with all empty frames)
        isec, nsec = 0, len(secondaries)
        for i in range(nextents):
            if extents[i]['index'] < 0:  # missing
                start = i * size
                name = basename + str(i)
                secondary = secondaries[isec]
                base_extpath = JavaSeisDataset.extent_dir(secondary, filename)
                path = osp.join(base_extpath, name)
                extents[i] = extent_dict(name, path, i, start, size)
                isec = 0 if isec == nsec-1 else isec + 1

        # the last extent might be a different size
        extent = extents[nextents-1]
        extent['size'] = maxpos - extent['start']

        return extents

    @staticmethod
    def extent_dir(secondary, filename):
        """
        -i- secondary : string, path of JavaSeis secondary disk
        -i- filename : string, JavaSeis dataset name
        -o- extdir : string, the JavaSeis dataset secondary directory
        """
        is_relative = osp.isabs(filename) == False
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
            if is_relative and not os.getcwd().startswith(datahome):
                raise ValueError(message1)
            raise ValueError(message2)
        elif is_relative:
            return osp.join(secondary, filename)
        else:
            pass # TODO joinpath(secondary, is_windows() ? filename[5:end] : filename[2:end])

# TODO
# read/write in parallel with multi thread


def parse_xml_file(filename):
    with open(filename, 'r') as f:
        data = f.read()
    root = etree.XML(data)
    return root


def get_extent_index(extents, offset):
    i = int(offset / extents[0]['size'])
    return extents[i]


def extent_dict(name="", path="", index=-1, start=-1, size=-1):
    return {'name': name, 'path': path, 'index': index, 'start': start,
            'size': size}


def get_names(path, start):
    names = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.startswith(start) and not file.endswith('.xml'):
                names.append(file)
    return names


def make_directory(abspath, force=False):
    if osp.isdir(abspath):
        if force: # delete old and create new
            shutil.rmtree(abspath)
            os.mkdir(abspath)
        else:
            warnings.warn("Directory already exists: {}".format(abspath))
    else:
        os.mkdir(abspath)


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


if __name__ == '__main__':
    testpath = "/home/zhu/datasets/test.js"
    if not osp.exists(testpath):
        print("'{0}' dataset does not exists".format(testpath))
    jsDataset = JavaSeisDataset(testpath)
    print(jsDataset)
