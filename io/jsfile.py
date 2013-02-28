import os
import dircache

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
JS_TRACE_HEADERS_XML "TraceHeaders.xml"

# other constants
JS_EARLYVERSION1 = "2006.01"
JS_PREXMLVERSION = "2006.2"
JS_VERSION = "2006.3"

def open_javaseis(filename):
    if not os.path.isdir(filename):
        raise IOError("JavaSeis dataset does not exists")

    if not os.access(filename, os.R_OK):
        raise IOError("Missing read access for JavaSeis dataset")

    js_dataset = new JavaSeisDataset(filename)

class JavaSeisDataset(object):
    def __init__(self, path):
        if not os.path.isdir(path):
            raise IOError("Must be a folder: %s" % path)

        self._validate_js_dir(path)

    def _validate_js_dir(files):
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



class JSFileReader(object):
    def __init__(self, bufferSize=2097152):
        self._bufferSize = bufferSize
        self._traceBuffer = None
        self._headerBuffer = None
        self._traceMap = None

    def open(self, file_name):
