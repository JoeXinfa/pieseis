from lxml import etree


def str2bool(str):
    if str.lower() in ('true', 't', 'y', 'yes', '1'):
        return True
    return False

def parset_element_is_array(text):
    if text.count('\n')>1:
        return True
    return False

def parset_element_get_value(text):
    """
    Either return the text string as a list of strings, or
    if not a list, then simply return the text string
    """
    if parset_element_is_array(text):
        return text.split()
    return text.strip()

def parse_parset_element(element):
    if not element.get('type'):
        return (None, None)

    element_type = element.get('type')
    retval = None
    value = element.text
    if element_type == 'string':
        retval = parset_element_get_value(value)

    elif element_type == 'boolean':
        retval = parset_element_get_value(value)
        if isinstance(retval, list):
            retval = list(map(str2bool, retval))
        else:
            retval = str2bool(retval)

    elif element_type == 'int':
        retval = parset_element_get_value(value)
        if isinstance(retval, list):
            retval = list(map(int, retval))
        else:
            retval = int(retval)

    elif element_type == 'double' or element_type == 'float':
        retval = parset_element_get_value(value)
        if isinstance(retval, list):
            retval = list(map(float, retval))
        else:
            retval = float(retval)

    # python2 long is int64, does python3 int work here?
    elif element_type == 'long':
        retval = parset_element_get_value(value)
        if isinstance(retval, list):
            retval = list(map(int, retval))
        else:
            retval = int(retval)

    return (element_type, retval)


class Properties(object):
    def __init__(self, root, name):
        if not isinstance(root, etree._Element):
            raise Exception("Root argument must be of type etree._Element")
        self._root = root
        self._attributes = {}
        self._name = name

        self._parse_parset(self._root)

    def put(self, name, value, value_type=None):
        if not name in self._attributes:
            self._attributes[name] = {'value': value, 'type': value_type}

    def get(self, attribute_name):
        return self._attributes[attribute_name]

    def get_attr_value(self, attribute_name):
        return self.get(attribute_name).get('value')

    def get_attr_type(self, attribute_name):
        return self.get(attribute_name).get('type')

    def _parse_parset(self, parset, parent=None):
        childrens = list(parset)
        for child in childrens:
            if child.tag == 'par':
                value_type, value = parse_parset_element(child)
                if parent:
                    self._attributes[parent][child.get('name')] = {'value': value, 'type': value_type}
                else:
                    self._attributes[child.get('name')] = {'value': value, 'type': value_type}
            else:  # parset
                name = child.get('name')
                self._attributes[name] = {}
                self._parse_parset(child, name)

    def _parse(self):
        children_elements = list(self._root)
        for child in children_elements:
            value_type, value = parse_parset_element(child)

            self.put(child.get('name'), value, value_type)

    def __str__(self):
        attribs = ""
        for key in self._attributes.keys():
            v = self._attributes[key]
            if isinstance(v, list):
                v = ', '.join(v)
            attribs += key
            attribs += " -> %s" % v

        import pprint
        from cStringIO import StringIO
        stream_buffer = StringIO()
        pprint.pprint(self._attributes, stream=stream_buffer)
        stream_buffer.seek(0)
        return '<%s %s>' % (self._name, stream_buffer.read())


class TraceHeadersXML(Properties):

    def __init__(self, root):
        super(TraceHeadersXML, self).__init__(root, "TraceHeaders")

    @property
    def nr_extents(self):
        value = self._attributes['VFIO_MAXFILE']
        return value.get('value')

    @property
    def extent_name(self):
        value = self._attributes['VFIO_EXTNAME']
        return value.get('value')

    @property
    def extent_size(self):
        value = self._attributes['VFIO_EXTSIZE']
        return value.get('value')

    @property
    def extent_maxpos(self):
        value = self._attributes['VFIO_MAXPOS']
        return value.get('value')


class TraceFileXML(Properties):

    def __init__(self, root):
        super(TraceFileXML, self).__init__(root, "TraceFile")

    @property
    def nr_extents(self):
        value = self._attributes['VFIO_MAXFILE']
        return value.get('value')

    @property
    def extent_name(self):
        value = self._attributes['VFIO_EXTNAME']
        return value.get('value')

    @property
    def extent_size(self):
        value = self._attributes['VFIO_EXTSIZE']
        return value.get('value')

    @property
    def extent_maxpos(self):
        value = self._attributes['VFIO_MAXPOS']
        return value.get('value')


class VirtualFolders(Properties):

    def __init__(self, root):
        super(VirtualFolders, self).__init__(root, "VirtualFolders")

    @property
    def nr_directories(self):
        value = self._attributes['NDIR']
        return value.get('value')

    @property
    def secondary_folders(self):
        secondaries = []
        n = self.nr_directories
        for i in range(n):
            name = "FILESYSTEM-{}".format(i)
            value = self._attributes[name]
            value = value.get('value')
            path = value.split(',')[0]
            secondaries.append(path)
        return secondaries


class FileProperties(Properties):

    def __init__(self, root):
        super(FileProperties, self).__init__(root, "FileProperties")

    @property
    def nr_dimensions(self):
        value = self._attributes['DataDimensions']
        return value.get('value')

    @property
    def data_type(self):
        value = self._attributes['DataType']
        return value.get('value')

    @property
    def header_length_in_bytes(self):
        value = self._attributes['HeaderLengthBytes']
        return value.get('value')

    @property
    def javaseis_version(self):
        return self._attributes['JavaSeisVersion']['value']

    @property
    def logical_deltas(self):
        value = self._attributes['LogicalDeltas']
        return value.get('value')

    def get_logical_delta(self, dimension):
        return self.get_logical_deltas()[dimension]

    @property
    def logical_origins(self):
        value = self._attributes['LogicalOrigins']
        return value.get('value')

    def get_logical_origin(self, dimension):
        return self.get_logical_origins()[dimension]

    def is_mapped(self):
        value = self._attributes['Mapped']
        return value.get('value')

    @property
    def physical_deltas(self):
        value = self._attributes['PhysicalDeltas']
        return value.get('value')

    def get_physical_delta(self, dimension):
        return self.get_physical_deltas()[dimension]

    @property
    def physical_origins(self):
        value = self._attributes['PhysicalOrigins']
        return value.get('value')

    def get_physical_origin(self, dimension):
        return self.get_physical_origins()[dimension]

    @property
    def trace_format(self):
        value = self._attributes['TraceFormat']
        return value.get('value')

    @property
    def axis_labels(self):
        value = self._attributes['AxisLabels']
        return value.get('value')

    @property
    def axis_lengths(self):
        value = self._attributes['AxisLengths']
        return value.get('value')

    def get_axis_length(self, dimension):
        return self.get_axis_lengths()[dimension]

    @property
    def axis_units(self):
        value = self._attributes['AxisUnits']
        return value.get('value')

    @property
    def axis_domains(self):
        value = self._attributes['AxisDomains']
        return value.get('value')

    def get_axis_unit(self, dimension):
        return self.get_axis_units()[dimension]

    @property
    def byte_order(self):
        return self._attributes['ByteOrder']['value']

    @property
    def comments(self):
        return self._attributes['Comments']['value']


class TraceProperties(Properties):
    def __init__(self, root):
        super(TraceProperties, self).__init__(root, "TraceProperties")

        self._header_names = [self._attributes[x]['label']['value'] for x in self._attributes]
        self._trace_headers_cache = {}

        self._trace_headers = {}
        for header, attributes in self._attributes.items():
            self._trace_headers[header] = TraceHeader(attributes)

        total_bytes = 0
        for key, value in self._trace_headers.items():
            total_bytes += value.format_size
        self._total_bytes = total_bytes

    @property
    def header_names(self):
        return self._header_names

    def header_values(self, header_name):
        """Return the TraceHeader instance with the name 'header_name'"""
        if not header_name in self.header_names:
            print("No header with name {0} found.".format(header_name))
            return None

        if header_name not in self._trace_headers_cache:
            for header_entry in self._attributes:
                if self._attributes[header_entry]['label']['value'] == header_name:
                    attribute_entry = self._attributes[header_entry]
                    self._trace_headers_cache[header_name] = TraceHeader(attribute_entry)
                    break
        return self._trace_headers_cache[header_name]

    @property
    def total_bytes(self):
        """Return total number of bytes for all the headers"""
        return self._total_bytes


class CustomProperties(Properties):
    def __init__(self, root):
        super(CustomProperties, self).__init__(root, "CustomProperties")
        """
        +Parse 'FieldInstruments'
        +Parse 'GeomMatchesFlag'
        +Parse 'Geometry'
        """
        #self._field_instruments = FieldInstruments(self.get('FieldInstruments'))

        self.data_properties = []
        parsets = ["FieldInstruments", "Geometry", "extendedParmTable"]
        for key in self._attributes:
            if key not in parsets:
                name = key
                fmt = self._attributes[key]['type']
                value = self._attributes[key]['value']
                self.data_properties.append(DataProperty(name, fmt, value))

    @property
    def synthetic(self):
        value = self._attributes['Synthetic']
        return value.get('value')

    @property
    def secondary_key(self):
        value = self._attributes['SecondaryKey']
        return value.get('value')

    @property
    def geometry_matches_flag(self):
        return self.get_attr_value('GeomMatchesFlag')

    @property
    def primary_key(self):
        return self.get_attr_value('PrimaryKey')

    @property
    def primary_sort(self):
        return self.get_attr_value('PrimarySort')

    @property
    def trace_no_matches_flag(self):
        return self.get_attr_value('TraceNoMatchesFlag')

    @property
    def stacked(self):
        return self.get_attr_value('Stacked')

    @property
    def cookie(self):
        return self.get_attr_value('cookie')


    # TODO: Create nested properties for 'field_instruments'
    #   and 'geometry' that returns already created classes

    """
        <parset name="FieldInstruments">
      <par name="systemFormatCode" type="int"> 2139081118 </par>
      <par name="nAuxChannels" type="int"> 2139081118 </par>
      <par name="systemSerialNum" type="int"> 2139081118 </par>
      <par name="earlyGain" type="float"> 0.0 </par>
      <par name="systemDialinConst" type="float"> 3.4E38 </par>
      <par name="systemManCode" type="int"> 2139081118 </par>
      <par name="notchFiltFreq" type="float"> 0.0 </par>
      <par name="highcutFiltSlope" type="float"> 0.0 </par>
      <par name="lowcutFiltFreq" type="float"> 0.0 </par>
      <par name="preampGain" type="float"> 0.0 </par>
      <par name="notchFiltSlope" type="float"> 0.0 </par>
      <par name="gainMode" type="int"> 0 </par>
      <par name="originalSamprat" type="float"> 0.0 </par>
      <par name="highcutFiltFreq" type="float"> 0.0 </par>
      <par name="originalNumsmp" type="int"> 2001 </par>
      <par name="aaFiltFreq" type="float"> 0.0 </par>
      <par name="sourceType" type="int"> 2139081118 </par>
      <par name="dateRecorded" type="int"> 0 </par>
      <par name="lowcutFiltSlope" type="float"> 0.0 </par>
      <par name="aaFiltSlope" type="float"> 0.0 </par>
    </parset>
    """

    class FieldInstruments(object):
        def __init__(self, data):
            self._data = data

        def get(self, attribute_name):
            return self._data[attribute_name]

        def get_attr_value(self, attribute_name):
            return self.get(attribute_name).get('value')

        def get_attr_type(self, attribute_name):
            return self.get(attribute_name).get('type')

        @property
        def get_system_format_code(self):
            return self._data['systemFormatCode']

    @property
    def field_instruments(self):
        if not hasattr(self, '_field_instruments'):
          # TODO: Create the reference / object instance here
          self._field_instruments = FieldInstruments(self.get('FieldInstruments'))

        return self._field_instruments

    @property
    def geometry(self):
        self._geometry


    """
    <parset name="Geometry">
      <par name="minCdpExternal" type="int"> 0 </par>
      <par name="nOffsetBins" type="int"> 2139081118 </par>
      <par name="nCrosslinesExternal" type="int"> 0 </par>
      <par name="ntracesTotal" type="long"> 2139081118 </par>
      <par name="nCdps" type="int"> 1002 </par>
      <par name="maxSin" type="int"> 2139081118 </par>
      <par name="incChan" type="int"> 2139081118 </par>
      <par name="ySurfLoc1" type="double"> 3.3999999521443642E38 </par>
      <par name="offsetMax" type="float"> 3.4E38 </par>
      <par name="units" type="int"> 3 </par>
      <par name="incCdpExternal" type="int"> 0 </par>
      <par name="xXLine1End" type="float"> 0.0 </par>
      <par name="yILine1End" type="float"> 0.0 </par>
      <par name="marine" type="int"> 0 </par>
      <par name="dCdpILine" type="float"> 25.0 </par>
      <par name="nInlinesExternal" type="int"> 0 </par>
      <par name="maxSurfLoc" type="int"> 2139081118 </par>
      <par name="multiComp" type="int"> 0 </par>
      <par name="maxNtrSource" type="int"> 2139081118 </par>
      <par name="maxILine" type="int"> 1 </par>
      <par name="xILine1Start" type="float"> 0.0 </par>
      <par name="cdpsAssigned" type="int"> 0 </par>
      <par name="dCdpXLine" type="float"> 500.0 </par>
      <par name="nILines" type="int"> 2 </par>
      <par name="incOffsetBin" type="int"> 2139081118 </par>
      <par name="minCdp" type="int"> 1 </par>
      <par name="nSurfLocs" type="int"> 2139081118 </par>
      <par name="offsetBinDist" type="float"> 3.4E38 </par>
      <par name="maxXLine" type="int"> 500 </par>
      <par name="finalDatum" type="float"> 3.4E38 </par>
      <par name="yXLine1End" type="float"> 500.0 </par>
      <par name="geomAssigned" type="int"> 0 </par>
      <par name="yRef" type="double"> 0.0 </par>
      <par name="incCdp" type="int"> 1 </par>
      <par name="datumVel" type="float"> 3.4E38 </par>
      <par name="azimuth" type="double"> 90.0 </par>
      <par name="xSurfLoc1" type="double"> 3.3999999521443642E38 </par>
      <par name="nXLines" type="int"> 501 </par>
      <par name="nLiveGroups" type="int"> 2139081118 </par>
      <par name="incSurfLoc" type="int"> 2139081118 </par>
      <par name="maxNtrCdp" type="int"> 2139081118 </par>
      <par name="maxCdp" type="int"> 1002 </par>
      <par name="maxOffsetBin" type="int"> 2139081118 </par>
      <par name="maxNtrRec" type="int"> 2139081118 </par>
      <par name="minILine" type="int"> 0 </par>
      <par name="maxChan" type="int"> 2139081118 </par>
      <par name="minChan" type="int"> 2139081118 </par>
      <par name="xRef" type="double"> 0.0 </par>
      <par name="xILine1End" type="float"> 12500.0 </par>
      <par name="minOffsetBin" type="int"> 2139081118 </par>
      <par name="nLiveShots" type="int"> 2139081118 </par>
      <par name="minXLine" type="int"> 0 </par>
      <par name="minSurfLoc" type="int"> 2139081118 </par>
      <par name="threeD" type="int"> 1 </par>
      <par name="yILine1Start" type="float"> 0.0 </par>
    </parset>
    """


class TraceHeader(object):
    """Correspond to a TraceHeader entry from the FileProperties.xml file"""
    def __init__(self, val=None):
#        if not val:
#            raise Exception("Missing trace header value")
        if val is not None:
            self.init_from_parset(val)

    def init_from_parset(self, val):
        self.label = val['label']['value']
        self.description = val['description']['value']
        self.format = val['format']['value']
        self.element_count = val['elementCount']['value']
        self.byte_offset = val['byteOffset']['value']
        self.format_string_to_type()

    def init_from_given(self, label, description, fmt, count, offset):
        self.label = label
        self.description = description
        self.format = fmt
        self.element_count = count
        self.byte_offset = offset
        self.format_string_to_type()

    def format_string_to_type(self):
        if self.format == "INTEGER":
            self.format_type = 'int32'
            self.format_size = 4 # bytes
        elif self.format == "LONG":
            self.format_type = 'int64'
            self.format_size = 8 # bytes
        elif self.format == "FLOAT":
            self.format_type = 'float32'
            self.format_size = 4 # bytes
        elif self.format == "DOUBLE":
            self.format_type = 'float64'
            self.format_size = 8 # bytes
        elif self.format == "BYTESTRING":
            self.format_type = 'uint8'
            self.format_size = 1 # bytes
        else:
            raise ValueError("unrecognized format".format(self.format))

    def __eq__(self, other):
        if self.label == other.label:
            return True
        return False


class DataProperty(object):
    """
    Correspond to a CustomProperties entry from the FileProperties.xml file
    """
    def __init__(self, label, fmt, value):
        self.label = label
        self.format = fmt
        self.value = value
