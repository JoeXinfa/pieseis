# -*- coding    " utf-8 -*-
"""
"""

from .properties import TraceHeader

trace_type = {
    "live" : 1,
    "dead" : 2,
    "aux"  : 3
}


stock_domain = {
    "ALACRITY" :        "alacrity",
    "AMPLITUDE" :       "amplitude",
    "COHERENCE" :       "coherence",
    "DELTA" :           "delta",
    "DENSITY" :         "density",
    "DEPTH" :           "depth",
    "DIP" :             "dip",
    "ENVELOPE" :        "envelope",
    "EPSILON" :         "epsilon",
    "ETA" :             "eta",
    "FLEX_BINNED" :     "flex_binned",
    "FOLD" :            "fold",
    "FREQUENCY" :       "frequency",
    "IMPEDANCE" :       "impedence",
    "INCIDENCE_ANGLE" : "incidence_angle",
    "MODEL_TRANSFORM" : "model_transform",
    "ROTATION_ANGLE" :  "rotation_angle",
    "SEMBLANCE" :       "semblence",
    "SLOTH" :           "sloth",
    "SLOWNESS" :        "slowness",
    "SPACE" :           "space",
    "TIME" :            "time",
    "UNKNOWN" :         "unknown",
    "VELOCITY" :        "velocity",
    "VS" :              "vs",
    "VSVP" :            "vsvp",
    "WAVENUMBER" :      "wavenumber"
}


stock_unit = {
    "DEGREES" :      "degrees",
    "FEET" :         "feet",
    "FT" :           "ft",
    "HERTZ" :        "hertz",
    "HZ" :           "hz",
    "M" :            "m",
    "METERS" :       "meters",
    "MICROSEC" :     "microseconds",
    "MILLISECONDS" : "milliseconds",
    "MS" :           "ms",
    "MSEC" :         "msec",
    "SECONDS" :      "seconds",
    "S" :            "seconds",
    "NULL" :         "null",
    "UNKNOWN" :      "unknown"
}


stock_dtype = {
    "CMP" :        "CMP",
    "CUSTOM" :     "CUSTOM",
    "OFFSET_BIN" : "OFFSET_BIN",
    "RECEIVER" :   "RECEIVER",
    "SOURCE" :     "SOURCE",
    "STACK" :      "STACK",
    "UNKNOWN" :    "UNKNOWN"
}


stock_props = {
    "AMP_NORM" :  TraceHeader(values=("AMP_NORM", "Amplitude normalization factor",    "FLOAT",   1, 0)),
    "AOFFSET" :   TraceHeader(values=("AOFFSET",  "Absolute value of offset",          "FLOAT",   1, 0)),
    "CDP" :       TraceHeader(values=("CDP",      "CDP bin number",                    "INTEGER", 1, 0)),
    "CDP_ELEV" :  TraceHeader(values=("CDP_ELEV", "Elevation of CDP",                  "FLOAT",   1, 0)),
    "CDP_NFLD" :  TraceHeader(values=("CDP_NFLD", "Number of traces in CDP bin",       "INTEGER", 1, 0)),
    "CDP_SLOC" :  TraceHeader(values=("CDP_SLOC", "External CDP number",               "INTEGER", 1, 0)),
    "CDP_X" :     TraceHeader(values=("CDP_X",    "X coordinate of CDP (float)",       "FLOAT",   1, 0)),
    "CDP_XD" :    TraceHeader(values=("CDP_XD",   "X coordinate of CDP (double)",      "DOUBLE",  1, 0)),
    "CDP_Y" :     TraceHeader(values=("CDP_Y",    "Y coordinate of CDP (float)",       "FLOAT",   1, 0)),
    "CDP_YD" :    TraceHeader(values=("CDP_YD",   "Y coordinate of CDP (double)",      "DOUBLE",  1, 0)),
    "CHAN" :      TraceHeader(values=("CHAN",     "Recording channel number",          "INTEGER", 1, 0)),
    "CMP_X" :     TraceHeader(values=("CMP_X",    "Average of shot and receiver x",    "FLOAT",   1, 0)),
    "CMP_Y" :     TraceHeader(values=("CMP_Y",    "Average of shot and receiver y",    "FLOAT",   1, 0)),
    "CR_STAT" :   TraceHeader(values=("CR_STAT",  "Corr. autostatics receiver static", "FLOAT",   1, 0)),
    "CS_STAT" :   TraceHeader(values=("CS_STAT",  "Corr. autostatics source static",   "FLOAT",   1, 0)),
    "DEPTH" :     TraceHeader(values=("DEPTH",    "Source depth",                      "FLOAT",   1, 0)),
    "DISKITER" :  TraceHeader(values=("DISKITER", "Disk data iteration*",              "INTEGER", 1, 0)),
    "DMOOFF" :    TraceHeader(values=("DMOOFF",   "Offset bin for DMO",                "INTEGER", 1, 0)),
    "DS_SEQNO" :  TraceHeader(values=("DS_SEQNO", "Input dataset sequence number*",    "INTEGER", 1, 0)),
    "END_ENS" :   TraceHeader(values=("END_ENS",  "End-of-ensemble flag*",             "INTEGER", 1, 0)),
    "END_VOL" :   TraceHeader(values=("END_VOL",  "End-of-volume flag*",               "INTEGER", 1, 0)),
    "EOJ" :       TraceHeader(values=("EOJ",      "End of job flag*",                  "INTEGER", 1, 0)),
    "FB_PICK" :   TraceHeader(values=("FB_PICK",  "First break pick",                  "FLOAT",   1, 0)),
    "FFID" :      TraceHeader(values=("FFID",     "Field file ID number",              "INTEGER", 1, 0)),
    "FILE_NO" :   TraceHeader(values=("FILE_NO",  "Sequential file number",            "INTEGER", 1, 0)),
    "FK_WAVEL" :  TraceHeader(values=("FK_WAVEL", "Wavelength of F-K domain trace",    "FLOAT",   1, 0)),
    "FK_WAVEN" :  TraceHeader(values=("FK_WAVEN", "Wavenumber of F-K domain trace",    "FLOAT",   1, 0)),
    "FNL_STAT" :  TraceHeader(values=("FNL_STAT", "Static to move to final datum",     "FLOAT",   1, 0)),
    "FRAME" :     TraceHeader(values=("FRAME",    "Frame index in framework",          "INTEGER", 1, 0)),
    "FT_FREQ" :   TraceHeader(values=("FT_FREQ",  "Frequency of F-T domain trace",     "FLOAT",   1, 0)),
    "GEO_COMP" :  TraceHeader(values=("GEO_COMP", "Geophone component (x,y,z)",        "INTEGER", 1, 0)),
    "HYPRCUBE" :  TraceHeader(values=("HYPRCUBE", "Hypercube index in framework",      "INTEGER", 1, 0)),
    "IF_FLAG" :   TraceHeader(values=("IF_FLAG",  "ProMax IF_FLAG",                    "INTEGER", 1, 0)),
    "ILINE_NO" :  TraceHeader(values=("ILINE_NO", "3D iline number",                   "INTEGER", 1, 0)),
    "LEN_SURG" :  TraceHeader(values=("LEN_SURG", "Length of surgical mute taper",     "FLOAT",   1, 0)),
    "LINE_NO" :   TraceHeader(values=("LINE_NO",  "Line number (hased line name)*",    "INTEGER", 1, 0)),
    "LSEG_END" :  TraceHeader(values=("LSEG_END", "Line segment end*",                 "INTEGER", 1, 0)),
    "LSEG_SEQ" :  TraceHeader(values=("LSEG_SEQ", "Line segment sequence number*",     "INTEGER", 1, 0)),
    "NA_STAT" :   TraceHeader(values=("NA_STAT",  "Portion of static not applied",     "FLOAT",   1, 0)),
    "NCHANS" :    TraceHeader(values=("NCHANS",   "Number of channels of source",      "INTEGER", 1, 0)),
    "NDATUM" :    TraceHeader(values=("NDATUM",   "Floating NMO Datum",                "FLOAT",   1, 0)),
    "NMO_STAT" :  TraceHeader(values=("NMO_STAT", "NMO datum static",                  "FLOAT",   1, 0)),
    "NMO_APLD" :  TraceHeader(values=("NMO_APLD", "NMO applied to traces",             "INTEGER", 1, 0)),
    "OFB_CNTR" :  TraceHeader(values=("OFB_CNTR", "Offset bin center",                 "FLOAT",   1, 0)),
    "OFB_NO" :    TraceHeader(values=("OFB_NO",   "Offset bin number",                 "INTEGER", 1, 0)),
    "OFFSET" :    TraceHeader(values=("OFFSET",   "Signed source-receiver offset",     "FLOAT",   1, 0)),
    "PAD_TRC" :   TraceHeader(values=("PAD_TRC",  "Artifically padded trace",          "INTEGER", 1, 0)),
    "PR_STAT" :   TraceHeader(values=("PR_STAT",  "Power autostatics receiver static", "FLOAT",   1, 0)),
    "PS_STAT" :   TraceHeader(values=("PS_STAT",  "Power autostatics source static",   "FLOAT",   1, 0)),
    "R_LINE" :    TraceHeader(values=("R_LINE",   "Receiver line number",              "INTEGER", 1, 0)),
    "REC_ELEV" :  TraceHeader(values=("REC_ELEV", "Receiver elevation",                "FLOAT",   1, 0)),
    "REC_H2OD" :  TraceHeader(values=("REC_H2OD", "Water depth at receiver",           "FLOAT",   1, 0)),
    "REC_NFLD" :  TraceHeader(values=("REC_NFLD", "Receiver fold",                     "INTEGER", 1, 0)),
    "REC_SLOC" :  TraceHeader(values=("REC_SLOC", "Receiver index number (internal)*", "INTEGER", 1, 0)),
    "REC_STAT" :  TraceHeader(values=("REC_STAT", "Total static for receiver",         "FLOAT",   1, 0)),
    "REC_X" :     TraceHeader(values=("REC_X",    "Receiver X coordinate (float)",     "FLOAT",   1, 0)),
    "REC_XD" :    TraceHeader(values=("REC_XD",   "Receiver X coordinate (double)",    "DOUBLE",  1, 0)),
    "REC_Y" :     TraceHeader(values=("REC_Y",    "Receiver Y coordinate (float)",     "FLOAT",   1, 0)),
    "REC_YD" :    TraceHeader(values=("REC_YD",   "Receiver Y coordinate (double)",    "DOUBLE",  1, 0)),
    "REPEAT" :    TraceHeader(values=("REPEAT",   "Repeated data copy number",         "INTEGER", 1, 0)),
    "S_LINE" :    TraceHeader(values=("S_LINE",   "Swath or sail line number",         "INTEGER", 1, 0)),
    "SAMPLE" :    TraceHeader(values=("SAMPLE",   "Sample index in framework",         "INTEGER", 1, 0)),
    "SEQNO" :     TraceHeader(values=("SEQNO",    "Sequence number in ensemble",       "INTEGER", 1, 0)),
    "SEQ_DISK" :  TraceHeader(values=("SEQ_DISK", "Trace sequence number from disk",   "INTEGER", 1, 0)),
    "SG_CDP" :    TraceHeader(values=("SG_CDP",   "Super gather CDP number",           "INTEGER", 1, 0)),
    "SIN" :       TraceHeader(values=("SIN",      "Source index number*",              "INTEGER", 1, 0)),
    "SLC_TIME" :  TraceHeader(values=("SLC_TIME", "Time slice input",                  "FLOAT",   1, 0)),
    "SMH_CDP" :   TraceHeader(values=("SMH_CDP",  "Number of CDP's in supergather",    "INTEGER", 1, 0)),
    "SOU_COMP" :  TraceHeader(values=("SOU_COMP", "Source component (x,y,z)",          "INTEGER", 1, 0)),
    "SOU_ELEV" :  TraceHeader(values=("SOU_ELEV", "Source elevation",                  "FLOAT",   1, 0)),
    "SOU_H2OD" :  TraceHeader(values=("SOU_H2OD", "Water depth at source",             "FLOAT",   1, 0)),
    "SOU_SLOC" :  TraceHeader(values=("SOU_SLOC", "External source location number",   "INTEGER", 1, 0)),
    "SOU_STAT" :  TraceHeader(values=("SOU_STAT", "Total static for source",           "FLOAT",   1, 0)),
    "SOU_X" :     TraceHeader(values=("SOU_X",    "Source X coordinate (float)",       "FLOAT",   1, 0)),
    "SOU_XD" :    TraceHeader(values=("SOU_XD",   "Source X coordinate (double)",      "DOUBLE",  1, 0)),
    "SOU_Y" :     TraceHeader(values=("SOU_Y",    "Source Y coordinate (float)",       "FLOAT",   1, 0)),
    "SOU_YD" :    TraceHeader(values=("SOU_YD",   "Source Y coordinate (double)",      "DOUBLE",  1, 0)),
    "SOURCE" :    TraceHeader(values=("SOURCE",   "Live source number (user-defined)", "INTEGER", 1, 0)),
    "SKEWSTAT" :  TraceHeader(values=("SKEWSTAT", "Multiplex skew static",             "FLOAT",   1, 0)),
    "SR_AZIM" :   TraceHeader(values=("SR_AZIM",  "Source to receiver azimuth",        "FLOAT",   1, 0)),
    "SRF_SLOC" :  TraceHeader(values=("SRF_SLOC", "External receiver location number", "INTEGER", 1, 0)),
    "TFULL_E" :   TraceHeader(values=("TFULL_E",  "End time of full samples",          "FLOAT",   1, 0)),
    "TFULL_S" :   TraceHeader(values=("TFULL_S",  "Start time of full samples",        "FLOAT",   1, 0)),
    "TIME_IND" :  TraceHeader(values=("TIME_IND", "Time sample index",                 "INTEGER", 1, 0)),
    "TLIVE_E" :   TraceHeader(values=("TLIVE_E",  "End time of live samples",          "FLOAT",   1, 0)),
    "TLIVE_S" :   TraceHeader(values=("TLIVE_S",  "Start time of live samples",        "FLOAT",   1, 0)),
    "TOT_STAT" :  TraceHeader(values=("TOT_STAT", "Total static for this trace",       "FLOAT",   1, 0)),
    "TRACE" :     TraceHeader(values=("TRACE",    "Trace index in framework",          "INTEGER", 1, 0)),
    "TRC_TYPE" :  TraceHeader(values=("TRC_TYPE", "Trace type (data, aux, etc.)",      "INTEGER", 1, 0)),
    "TR_FOLD" :   TraceHeader(values=("TR_FOLD",  "Actual trace fold",                 "FLOAT",   1, 0)),
    "TRACENO" :   TraceHeader(values=("TRACENO",  "Trace number in seismic line*",     "INTEGER", 1, 0)),
    "TRIMSTAT" :  TraceHeader(values=("TRIMSTAT", "Trim static",                       "FLOAT",   1, 0)),
    "UPHOLE" :    TraceHeader(values=("UPHOLE",   "Source uphole time",                "FLOAT",   1, 0)),
    "VOLUME" :    TraceHeader(values=("VOLUME",   "Volume index in framework",         "INTEGER", 1, 0)),
    "WB_TIME" :   TraceHeader(values=("WB_TIME",  "Water bottom time",                 "FLOAT",   1, 0)),
    "XLINE_NO" :  TraceHeader(values=("XLINE_NO", "3D crossline number",               "INTEGER", 1, 0))
}


# from promax manual, this is the minimal (guaranteed) set of properties
minimal_props = [
    "SEQNO",
    "END_ENS",
    "EOJ",
    "TRACENO",
    "TRC_TYPE",
    "TLIVE_S",
    "TFULL_S",
    "TFULL_E",
    "TLIVE_E",
    "LEN_SURG",
    "TOT_STAT",
    "NA_STAT",
    "AMP_NORM",
    "TR_FOLD",
    "SKEWSTAT",
    "LINE_NO",
    "LSEG_END",
    "LSEG_SEQ"
]
