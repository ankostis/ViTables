#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#       Copyright (C) 2005-2007 Carabos Coop. V. All rights reserved
#       Copyright (C) 2008-2013 Vicent Mas. All rights reserved
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
from pandas.io.pytables import HDFStore, _tables
from tables.file import File


__docformat__ = 'restructuredtext'


class HDFStoreWrapper(HDFStore):

    """Subclassed to wrap `h5py.File` in instances"""

    def __init__(self, tables_file: File):
        self._path = tables_file.filename
        self._mode = tables_file.mode
        self._handle = tables_file
        self._filters = tables_file.filters

        _tables()
