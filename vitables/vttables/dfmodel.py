#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#       Copyright (C) 2005-2007 Carabos Coop. V. All rights reserved
#       Copyright (C) 2008-2017 Vicent Mas. All rights reserved
#
#       This program is free software: you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation, either version 3 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#       Author:  Vicent Mas - vmas@vitables.org
"""
This module implements a model (in the `MVC` sense) for the real data stored
in a `tables.Leaf`.
"""

__docformat__ = 'restructuredtext'

import logging
import vitables.utils

from qtpy import QtCore, QtGui
from qtpy.QtCore import Qt

logger = logging.getLogger(__name__)

_axis_font = QtGui.QFont()
_axis_font.setBold(True)

_axis_label_font = QtGui.QFont()
_axis_label_font.setBold(True)
_axis_label_font.setItalic(True)


def try_as_pandas_dataframe(leaf):
    try:
        from pandas.io import pytables
    except ImportError:
        return

    class HDFStoreWrapper(pytables.HDFStore):
        """Subclassed to construct stores from :class:`h5py.File` instances."""

        def __init__(self, tables_h5file):
            self._path = tables_h5file.filename
            self._mode = tables_h5file.mode
            self._handle = tables_h5file
            self._filters = tables_h5file.filters

            pytables._tables()

    pgroup = leaf._g_getparent()
    assert pgroup._c_classid == 'GROUP', (leaf, pgroup)

    pandas_attr = getattr(pgroup._v_attrs, 'pandas_type', None)
    if pandas_attr == 'frame_table':  # TODO: Support also other 'fixed' format.
        hstore = HDFStoreWrapper(leaf._v_file)

        return DataFrameModel(leaf, hstore)


class DataFrameModel(QtCore.QAbstractTableModel):
    """
    The model for data contained in pandas DataFrame chunks.

    :Parameters:

        - `rbuffer`: a buffer used for optimizing read access to data
        - `parent`: the parent of the model
    """

    #: How many rows to load and show in the qt-table when scrolling.
    chunk_size = 1000

    def __init__(self, leaf, hstore, parent=None):
        """Create the model.
        """
        import pandas as pd

        super(DataFrameModel, self).__init__(parent)

        self._leaf = leaf
        self._hstore = hstore
        self.start = 0

        ## The dataset number of rows is potentially huge but tables are
        #  kept small: just the data returned by a read operation of the
        #  buffer are displayed
        self.nrows, self.ncols = leaf.shape

        # Track selected cell.
        self.selected_cell = {'index': QtCore.QModelIndex(), 'buffer_start': 0}

        # Populate the model with the first chunk of data.
        self._chunk = chunk = self.loadData(0, self.chunk_size)

        def count_multiindex(index):
            if isinstance(index, pd.MultiIndex):
                return len(index.levels)
            return 1

        self._nheaders_y, self._nheaders_x = [count_multiindex(x)
                                              for x
                                              in [chunk.index, chunk.columns]]

    def headerData(self, section, orientation, role):
        """
        Model method to return header content and formatting.

        :param section:
            The zero-based row/column number to return.
        :param orientation:
            The header orientation (horizontal := columns, vertical := index).
        :param role:
            The Qt.XXXRole: being inspected.
        """
        if role == Qt.TextAlignmentRole:
            if orientation == Qt.Horizontal:
                return Qt.AlignCenter | Qt.AlignBottom
            return Qt.AlignRight | Qt.AlignVCenter

        if role != Qt.DisplayRole:
            return None

        return str(section)

    def data(self, index, role=Qt.DisplayRole):
        """Returns the data stored under the given role for the item
        referred to by the index.

        This is an overwritten method.

        :Parameters:

        - `index`: the index of a data item
        - `role`: the role being returned
        """
        row, col = index.row(), index.column()
        if not index.isValid() or not (0 <= row < self.nrows):
            return None

        nhy, nhx = self._nheaders_y, self._nheaders_x
        is_hy = row <= nhy
        is_hx = row <= nhx

        if is_hy and is_hx:
            if role == Qt.DisplayRole:
                return 'RC'  # TODO: print index labels
            if role == Qt.FontRole:
                return _axis_label_font
            return None

        if is_hy:
            if role == Qt.DisplayRole:
                return 'row %s' % row # XXX: index
            if role == Qt.FontRole:
                return _axis_font
            if role == Qt.TextAlignmentRole:
                return int(Qt.AlignRight | Qt.AlignVCenter)
            return None

        if is_hx:
            if role == Qt.DisplayRole:
                return 'col %s' % col # XXX: index
            if role == Qt.FontRole:
                return _axis_font
            if role == Qt.TextAlignmentRole:
                return int(Qt.AlignCenter | Qt.AlignBottom)
            return None

        cell = self.chunk.iat[row - self.start - nhy, col - nhx]
        if role == Qt.DisplayRole:
            return self.formatContent(cell)
#        if role == Qt.TextAlignmentRole:
#            return int(Qt.AlignLeft|Qt.AlignTop)
        return None

    def columnCount(self, index=QtCore.QModelIndex()):
        """
        The total number of columns, or number of children for the given index.

        When implementing a table based model this method should return 0
        for valid indices (because they have no children) or the number of
        the total *columns* exposed by the model.

        :param index:
            The model index being inspected.
        """
        return 0 if index.isValid() else self.ncols

    def rowCount(self, index=QtCore.QModelIndex()):
        """
        The total number of rows, or number of children for the given index.

        When implementing a table based model this method should return 0
        for valid indices (because they have no children) or the number of
        the total *rows* exposed by the model.

        :param index:
            The model index being inspected.
        """

        return 0 if index.isValid() else self.nrows

    def loadData(self, start, length):
        """Load the model with fresh data from the buffer.

        :param start:
            the the first row of the chunk
        :param length:
            the number of rows to be read.
        """
        ## Enforce scroll limits.
        #
        start = max(start, 0)
        stop = min(start + length, self.nrows)
        assert stop >= start, (start, stop, length)

        ## Ensure that the whole buffer will be filled (correct ??).
        #
        actual_start = stop - self.chunk_size
        start = max(min(actual_start, start), 0)

        pgroup = self._leaf._g_getparent()
        return self._hstore.select(pgroup._v_pathname, start=start, stop=stop)
