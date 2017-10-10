#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2017-10-06 at 13:29

@author: cook



Version 0.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from tqdm import tqdm
import warnings
import os
import time
import pkg_resources

# =============================================================================
# Define variables
# =============================================================================
TEMPLATE_FILE = 'template.txt'
LOGLEVEL = 'all'
__PACKAGE__ = "MachineReadableTable"  # Could be any module/package name



# -----------------------------------------------------------------------------

# =============================================================================
# Define functions
# =============================================================================
def logger(kind, message):
    levels = dict(all=0, info=1, warn=2, error=3)
    mytime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    if LOGLEVEL == 'all':
        log = True
    elif levels[kind] <= levels[LOGLEVEL]:
        log = True
    else:
        log = False
    if log:
        print('|--{0}--| {1} | -- | {2}'.format(kind.upper(), mytime, message))


class MachineReadableTable:

    def __init__(self, title=None, authors=None, tablename=None, filename=None,
                 labels=None, descs=None, units=None, formats=None):
        """

        :param title: string, the title of the table
        :param authors: string, the authors of the table
        :param tablename: string, the name of the table
        :param filename: string, the filename of the input data file
        :param labels: list of strings, the desired names for the columns
        :param descs: list of strings, the desired descriptions for the columns
        :param units: list of strings, the desired units for the columns
        :param formats: list of strings, the desired formats for the columns
        """
        # set variables
        self.Ncols = 0
        self.title = title
        self.authors = authors
        self.name = tablename
        self.filename = filename
        self.labels = labels
        self.descs = descs
        self.units = units
        self.formats = formats
        self.strbytes, self.sbytes, self.ebytes = [], [], []
        self.rawdata = None
        self.data, self.bytes = '', ''

    def make_table(self, out=None):

        # verify inputs
        self.__variable_verification()

        # deal with data and byte acquisition
        self.__get_data_bytes()

        # Get the template file name
        template_file = pkg_resources.resource_filename(__PACKAGE__,
                                                        TEMPLATE_FILE)
        # Read template file (as string)
        self.template = read_txt_file_into_string(template_file)

        ffilename = self.filename.split('/')[-1]
        # Replace strings in template
        self.template = self.template.replace('$title', self.title)
        self.template = self.template.replace('$authors', self.authors)
        self.template = self.template.replace('$tablename', self.name)
        self.template = self.template.replace('$filename', ffilename)
        self.template = self.template.replace('$bytes', self.bytes)
        self.template = self.template.replace('$data', self.data)

        if out == 'print':
            print(self.template)
        elif type(out) == str:
            f = open(out, 'w')
            f.write(self.template)
            f.close()
        else:
            pass

    def __variable_verification(self):
        """
        Make sure variables inputted are in correct format/length

        :param title: string, the title of the table
        :param authors: string, the authors of the table
        :param tablename: string, the name of the table
        :param filename: string, the filename of the input data file
        :param labels: list of strings, the desired names for the columns
        :param descs: list of strings, the desired descriptions for the columns
        :param units: list of strings, the desired units for the columns
        :param formats: list of strings, the desired formats for the columns
        :return:
        """
        # check strings
        vars = [self.title, self.authors, self.name, self.filename]
        names = ['title', 'authors', 'tablename', 'filename']
        for v, var in enumerate(vars):
            if type(var) != str:
                emsg = ('"{0}" must be a valid string.\n'.format(names[v]))
                raise ValueError(emsg)

        # check that file exisself.ts
        if not os.path.exists(self.filename):
            emsg = 'File "{0}" does not exists'.format(self.filename)
            raise FileNotFoundError(emsg)
        # check that labels/descs/units/formats are None or the same length
        n, defined = None, []
        vars = [self.labels, self.descs, self.units, self.formats]
        names = ['labels', 'descs', 'units', 'formats']
        for v, var in enumerate(vars):
            if var is not None:
                if len(var) != n and n is not None:
                    emsg = '{0} must be "None" or same length as {1}'
                    raise ValueError(emsg.format(' and '.join(defined)))
                else:
                    n = len(var)
                    defined.append(names[v])
        self.Ncols = n

    def __get_data_bytes(self):
        # Deal with the data
        logger("info", "Formatting data values...")
        self.get_data()
        # Deal with the column names
        logger("info", "Getting column labels...")
        self.get_cols()
        # Deal with the column formats
        logger("info", "Getting column formats...")
        self.get_formats()
        # Deal with the column descriptions
        logger("info", "Getting column descriptions...")
        self.get_descriptions()
        # Deal with the column units
        logger("info", "Getting column units...")
        self.get_units()
        # Deal with the bytes construction
        logger("info", "Constructing byte-by-byte table...")
        self.construct_bytes()

    def get_data(self):
        # load data
        with warnings.catch_warnings(record=True) as w:
            self.rawdata = Table.read(self.filename)
        # save tmp file in correct format
        self.rawdata.write('tmp.txt', format='ascii.fixed_width_no_header',
                           overwrite=True)
        # load tmp.txt into self.data variable
        output = read_txt_file_into_string('tmp.txt')
        # remove additional unneeded formatting
        remove = [r"b'", "'", '"', "| ", "|"]
        for r in remove:
            output = output.replace(r, '')
        self.data = output
        # delete tmp.txt
        os.remove('tmp.txt')

    def get_cols(self):
        # if labels is None then get the column names from the data file
        if self.labels is None:
            self.labels = self.rawdata.colnames
        # else check that the labels/units/format/descriptions are the same
        # length as the data
        else:
            if self.Ncols != len(self.rawdata.colnames):
                emsg = ("Number of columns in data (={0}) and length of labels"
                        "/units/formats/description (={1}) do not match!")
                raise ValueError(emsg.format(len(self.rawdata.colnames),
                                             self.Ncols))
        # Now we need to check that columns are okay:
        # Check 1: No spaces
        # Check 2: charaters $ @ ! ` \ ^ ~ & are not used
        # Check 3: _ is preceeded only by a d E e f l m n o q r u w x
        # Check 4: all _ variables have an assiociated non _ column
        # Check 5: No column name is used twice
        check_columns(self.labels)

        # Check 6: check for '_' in any col
        # if there are '_' columns then format columns
        has_under = False
        for label in self.labels:
            if '_' in label:
                has_under = True
        if has_under:
            new_labels = []
            for label in self.labels:
                if '_' not in label:
                    new_labels.append('  ' + label)
                else:
                    new_labels.append(label)
            self.labels = list(new_labels)

    def get_formats(self):
        # if formats is None then get formats from raw data
        if self.formats is None:
            formats = []
            for col in self.rawdata.colnames:
                formats.append(get_fortran_format(self.rawdata[col]))
                formats.append('X1')
            self.formats = formats
        else:
            self.formats = list(self.formats)
        # Need to check the formats
        # Check 1: in A I E F X
        # Check 2: form An In En.m Fn.m Xn
        check_formats(self.formats, self.labels)

    def get_descriptions(self):
        # if descriptions is None then get from raw data
        if self.descs is None:
            descriptions = []
            for col in self.rawdata.colnames:
                # get description
                desc = self.rawdata[col].description
                descriptions.append(desc)
        else:
            descriptions = list(self.descs)
        # reformat descriptions
        for d_it, col in enumerate(self.rawdata.colnames):
            desc = descriptions[d_it]
            # take out double spaces
            while '  ' in desc:
                desc = desc.replace('  ', ' ')
            # Remove any \n \t from descriptions
            desc = desc.replace('\n', '')
            desc = desc.replace('\t', '')
            # Remove any current ? from start of descriptions
            if desc[0] == '?':
                desc = desc[1:]
            # figure out if we have one or more blank rows (NaN values)
            # if we do there should be a ? at the start
            try:
                values = np.array(self.rawdata[col], dtype=float)
                if len(values) != np.sum(np.isfinite(values)):
                    desc = '? ' + desc
            except ValueError:
                pass
            descriptions[d_it] = desc

        self.descs = descriptions

    def get_units(self):
        if self.units is None:
            units = []
            for col in self.rawdata.colnames:
                unit = self.rawdata[col].unit
                units.append(unit)
        else:
            units = list(self.units)
        check_units(units, self.labels)
        # convert units to strings passing through astropy units
        nunits = []
        buffer = ''
        eformat = '\t\tColumn {0}: name="{1}": {2}\n'
        for u_it, unit in enumerate(units):
            if (unit is None) or unit == '---':
                newunit = '---'
            else:
                newunit = str(u.Unit(str(unit)))
            # remove white spaces
            newunit = newunit.replace(' ', '')
            # print consistency
            if (newunit != str(unit)) and (unit is not None):
                emsg = ('{0} --> {1}'.format(unit, newunit))
                buffer += eformat.format(u_it, self.labels[u_it], emsg)
            # append to new list
            nunits.append(newunit)
        self.units = list(nunits)
        # Summary of changes
        logger('info', '\tSummary of unit changes:\n\n' + buffer)

    def construct_bytes(self):
        self.get_bytes()
        # create byte table
        self.bytes = self.construct_byte_table()

    def construct_byte_table(self):

        kformat = kept_formats(self.formats)

        vars = [self.strbytes, kformat, self.units, self.labels, self.descs]
        # Get the maximum length of each column
        lens = []
        for var in vars:
            vlens = []
            for rowv in var:
                vlens.append(len(str(rowv)))
            lens.append(np.max(vlens))
        # add the rows
        table = ''
        for row in range(len(self.strbytes)):
            arg = [self.strbytes[row], kformat[row], self.units[row],
                   self.labels[row], self.descs[row]]
            for li, leni in enumerate(lens):
                table += ('{:<' + str(leni) + '} ').format(arg[li])
            table += '\n'
        return table

    def get_bytes(self):
        self.sbytes = []
        startlen = 0
        self.ebytes = []
        endlen = 0
        running = 1
        for fmt in self.formats:
            # get the byte from fmt
            kind = fmt[0]
            raw = fmt[1:].split('.')[0]
            byte = int(raw)
            # if kind is 'X' then add skip
            if kind == 'X':
                running += byte
                continue
            # set the start to "running"
            self.sbytes.append(running)
            if len(str(running)) > startlen:
                startlen = len(str(running))
            # set the end to "running" + byte - 1
            end = running + byte - 1
            self.ebytes.append(end)
            if len(str(end)) > endlen:
                endlen = len(str(end))
            # finally increment running
            running += byte
        # construct the string byte
        self.strbytes = []
        for s_it in range(len(self.sbytes)):
            stringbyte = ('{:' + str(startlen) + '.0f}').format(self.sbytes[s_it])
            stringbyte += ('-{:' + str(endlen) + '.0f}').format(self.ebytes[s_it])
            self.strbytes.append(stringbyte)

    def __set_prop(self, var, name, kind=str):
        if type(var) != kind:
            emsg = ('{0} must be a valid {1}.\n'.format(name, kind))
            raise ValueError(emsg)
        return 1

    def set_title(self, title):
        self.__set_prop(title, 'Title')
        self.title = title

    def set_filename(self, filename):
        self.__set_prop(title, 'Filename')
        if not os.path.exists(filename):
            emsg = 'File "{0}" does not exists'.format(self.filename)
            raise FileNotFoundError(emsg)
        self.filename = filename

    def set_authors(self, authors):
        self.__set_prop(authors, 'Authors')
        self.authors = authors

    def set_tablename(self, tablename):
        self.__set_prop(tablename, 'Table name')
        self.name = tablename

    def set_labels(self, labels):
        self.__set_prop(labels, 'Labels', list)
        self.labels = labels

    def set_descs(self, descriptions):
        self.__set_prop(labels, 'Descriptions', list)
        self.descs = descriptions

    def set_units(self, units):
        self.__set_prop(units, 'Units', list)
        self.units = units

    def set_formats(self, formats):
        self.__set_prop(formats, 'Formats', list)
        self.formats = formats


def read_txt_file_into_string(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    string = ''
    for line in lines:
        string += line
    return string


def check_columns(labels):
    # Now we need to check that columns are okay:
    buffer = ''
    eformat = 'Column {0}: name="{1}" - {2}\n'
    for l_it, label in enumerate(labels):
        # Check 1: No spaces
        if ' ' in label:
            emsg = 'white spaces are not allowed'
            buffer += eformat.format(l_it, label, emsg)
        # Check 2: charaters $ @ ! ` \ ^ ~ & are not used
        for char in ['$', '@', '!', '`', '\\', '^', '~']:
            if char in label:
                emsg = 'character "{0}" are not allowed'.format(char)
                buffer += eformat.format(l_it, label, emsg)
        # Check 3: _ is preceeded only by a d E e f l m n o q r u w x
        if ('_' in label):
            passed = False
            for char in 'adEeflmnoqruwx':
                if char + '_' in label:
                    passed = True
            if not passed:
                emsg = (' "_" must be preceeded by a d E e f l m n o q r u w'
                        ' x only')
                buffer += eformat.format(l_it, label, emsg)
            # Check 4: all _ variables have an assiociated non _ column
            slabel = label.split('_')[0] + '_'
            rest = label.split(slabel)[-1]
            if rest not in labels:
                emsg = ('must have a column '
                        '"{0}" associated with it').format(rest)
                buffer += eformat.format(l_it, label, emsg)
        # Check 5: No column name is used twice
        if label in labels[:l_it] or label in labels[l_it + 1:]:
            emsg = 'must be a unique column name (found more than once)'
            buffer += eformat.format(l_it, label, emsg)

    if len(buffer) != 0:
        raise ValueError("Error in column names: \n" + buffer)


def check_formats(formats, labels):
    # Now we need to check that columns are okay:
    buffer = ''
    eformat = 'Column {0}: name="{1}" format="{2}" - {3}\n'

    for f_it, label in enumerate(labels):
        formati = formats[f_it].upper()
        # if format is a string or integer or skip must be in form Yn
        if formati[0] in list('AIX'):
            try:
                _ = int(formati[1:])
            except ValueError:
                emsg = ('invalid format must be {0}n    where n is an '
                        'integer').format(formati[0])
                buffer += eformat.format(f_it, label, formati, emsg)
        # if format is a float or scientific must be in form Yn.m or Yn.mEq
        elif formati[0] in list('EF'):
            try:
                _ = float(formati[1:])
            except ValueError:
                emsg = ('invalid format must be {0}n.m   where n and m '
                        'are integers').format(formati[0])
                buffer += eformat.format(f_it, label, formati, emsg)
        # else we need to record the error
        else:
            emsg = 'must start with an A I F E or X'
            buffer += eformat.format(f_it, label, formati, emsg)

    if len(buffer) != 0:
        raise ValueError("Error in column names: \n " + buffer)


def kept_formats(formats):
    new_formats = []
    for fmt in formats:
        if 'X' in fmt:
            continue
        else:
            new_formats.append(fmt)
    return new_formats


def check_units(units, labels):
    # Now we need to check that columns are okay:
    buffer = ''
    eformat = 'Column {0}: name="{1}" unit="{2}" - {3}\n'

    for u_it, label in enumerate(labels):
        unit = units[u_it]
        # set the table units
        try:
            unit = u.Unit(unit)
        except ValueError as emsg:
            if unit is None or unit == '---':
                pass
            else:
                buffer += eformat.format(u_it, label, unit, emsg)
        except TypeError as emsg:
            if unit is None or unit == '---':
                pass
            else:
                buffer += eformat.format(u_it, label, unit, emsg)

    if len(buffer) != 0:
        raise ValueError("Error in column names: \n " + buffer)


def get_fortran_format(x):
    tt = str(np.array(x).dtype)
    n, m = 1, 1
    science = False
    for xi in x:
        string = repr(xi).strip()
        if string == '--':
            continue
        if 'float' in tt:
            # deal with scientific notation
            strings1 = string.split('e')
            try:
                o0 = len(strings1[1])
                science = True
            except IndexError:
                o0 = 0
            n0 = 1 + len(strings1[0]) + o0
            # deal with decimal point
            strings2 = strings1[0].split('.')
            try:
                m0 = len(strings2[1])
            except IndexError:
                m0 = 0
            if m0 > m:
                m = m0
        else:
            n0 = len(string)
        if n0 > n:
            n = n0

    if 'int' in tt:
        return 'I{0}'.format(n)
    elif 'bool' in tt:
        return 'I1'
    elif 'float' in tt and science:
        return 'E{0}.{1}'.format(n, m)
    elif 'float' in tt:
        return 'F{0}.{1}'.format(n, m)
    else:
        return 'A{0}'.format(n)


# =============================================================================
# Start of code
# =============================================================================
# Main code here
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # Run test
    title = 'test readme'
    authors = 'Neil J Cook'
    tablename = 'test table 1'

    workspace = '/scratch/Projects/RayAleks/Data/final/not_cut_down'

    infilename = (workspace + '/L-ZYJHK.votable')
    outfile = (workspace + '/L-ZYJHK_README.txt')
    colfile = (workspace + '/full_Master_lsample.fits')

    infilename = (workspace + '/C-ZYJHK_DR10_noRed_noHK.votable')
    outfile = (workspace + '/C-ZYJHK_DR10_noRed_noHK_README.txt')
    colfile = (workspace + '/full_Master_csample.fits')

    # load column info
    logger('info', 'Opening column file...')
    coldata = Table.read(colfile)
    labels = list(coldata['Label'])

    # create machine readable table object
    mrt = MachineReadableTable(title, authors, tablename, infilename)
    mrt.set_labels(labels)
    # write table to file
    mrt.make_table(out=None)




# =============================================================================
# End of code
# =============================================================================
