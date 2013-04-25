import os
import dircache
from lxml import etree

from properties import FileProperties, TraceProperties, CustomProperties

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

def open_javaseis(path):
    if not os.path.isdir(path):
        raise IOError("JavaSeis dataset does not exists")

    if not os.access(path, os.R_OK):
        raise IOError("Missing read access for JavaSeis dataset")

    js_dataset = JavaSeisDataset(path)
    return js_dataset


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
        except IOError, ioe:
            print("%s is not a valid dataset" % path)
            self._is_valid = False

        self._files = dircache.listdir(path)
        self.path = path
        self._read_properties()


        # self.read_data()
        self._is_open = True


    def _validate_js_dir(self, path):
        """Gets called during the construction of this object instance"""
        def js_missing(f):
            raise IOError("Missing: "+f)
        files = dircache.listdir(path)

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

    def _read_properties(self):
        filename = os.path.join(self.path, JS_FILE_PROPERTIES_XML)

        data = None
        with open(filename, 'r') as f:
            data = f.read()

        root = etree.XML(data)
        if root.get('name') != 'JavaSeis Metadata':
            raise IOError(JS_FILE_PROPERTIES_XML+" is not a JavaSeis Metadata file!")

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

    def get_nr_dimension(self):
        return self._file_properties

    def is_valid(self):
        return self._is_valid


class JSFileReader(object):
    def __init__(self, bufferSize=2097152):
        self._bufferSize = bufferSize
        self._traceBuffer = None
        self._headerBuffer = None
        self._traceMap = None

    def open(self, path):
        js_dataset = open_javaseis(path)



if __name__ == '__main__':
    testpath = "/home/asbjorn/datasets/2hots.js"
    jsDataset = JavaSeisDataset(testpath)
    print jsDataset