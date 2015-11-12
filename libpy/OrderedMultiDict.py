# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

from __future__ import print_function

import sys
if sys.version_info.major >= 3:
    from configparser import RawConfigParser
else:
    from ConfigParser import RawConfigParser


class OrderedMultiDict(dict):
    """A ConfigParser-compatible dictionary that can remember multiple values
    per key.

    The tricky part is getting around this code at the end of
    `ConfigParser.RawConfigParser._read(...)`:

        # join the multi-line values collected while reading
        all_sections = [self._defaults]
        all_sections.extend(self._sections.values())
        for options in all_sections:
            for name, val in options.items():
                if isinstance(val, list):
                    options[name] = '\n'.join(val)

    Also relevant:  Observe that option-values are added to the dictionary
    as elements in a LIST before that:

        optname, vi, optval = mo.group('option', 'vi', 'value')
        <snip>
        if optval is not None:
            <snip>
            optval = optval.strip()
            # allow empty values
            if optval == '""':
                optval = ''
            cursect[optname] = [optval]
        else:
            # valueless option handling
            cursect[optname] = optval

    Containment of option-values as elements in a list is used to enable
    multi-line values:

        # continuation line?
        if line[0].isspace() and cursect is not None and optname:
            value = line.strip()
            if value:
                cursect[optname].append(value)

    We will work around this constraint by maintaining a separate list of
    key-value pairs, in insertion order.  We will pretend to RawConfigParser
    that we have only one value for each key, but we'll maintain a secret list
    of ALL key-value pairs inserted, over which we can iterate afterwards.

    Note also that RawConfigParser uses its one `dict_type` to store all three
    of these conceptually-distinct mappings:
     - section name -> section dict  (in the dict `_sections`)
     - defaults  (in the dict `_defaults`)
     - each section dict  (in each `_sections[sectname]` value)

    The methods/attributes that RawConfigParser expects its `dict_type` to have
    are:
     - __init__(self)
     - __contains__(self, key)  # ie, (key in d)
        https://docs.python.org/2/reference/datamodel.html#object.__contains__
     - __delitem(self, key)  # ie, del d[key]
        https://docs.python.org/2/reference/datamodel.html#object.__delitem__
     - __getitem__(self, key)  # ie, value_in_dict = d[key]
        https://docs.python.org/2/reference/datamodel.html#object.__getitem__
     - __setitem__(self, key, val)  # ie, d[key] = val
        https://docs.python.org/2/reference/datamodel.html#object.__setitem__
     - copy(self)
     - items(self)
     - keys(self)
     - update(self, other)
     - values(self)

    We'll make all of these methods/attributes pretend to RawConfigParser that
    this class is just a regular Python `dict`.  To access the "secret list",
    use the method `items(self)`.
    """
    def __init__(self, other={}):
        self._keyval_pairs = []
        self.counter = 0
        self.update(other)

    def __delitem__(self, key):
        self._keyval_pairs = [(k, v)
                for k, v in self._keyval_pairs if k.key != key]
        dict.__delitem__(self, key)

    def __setitem__(self, key, val):
        if isinstance(key, KeyWithInteger):
            # Replace the current value with this key.
            for i, (k, v) in enumerate(self._keyval_pairs):
                if k == key:
                    self._keyval_pairs[i] = (key, val)
                    dict.__setitem__(self, key.key, val)
                    return

            # This key is not an element of this dict.
            key = key.key

        # Insert a new value with this key.
        self.counter += 1
        key_with_integer = KeyWithInteger(key, self.counter)
        self._keyval_pairs.append((key_with_integer, val))
        dict.__setitem__(self, key, val)

    def copy(self):
        return self.__class__(self)

    def items(self):
        # Return a copy, so the naughty user can't edit it.
        return self._keyval_pairs[:]

    def update(self, other):
        if isinstance(other, OrderedMultiDict):
            for k, v in other.items():
                self[k.key] = v
        else:
            for k, v in other.items():
                self[k] = v

    def values(self):
        return [v for k, v in self._keyval_pairs]


class KeyWithInteger(object):
    def __init__(self, key, integer):
        self.key = key
        self.integer = integer

    def __eq__(self, other):
        if isinstance(other, KeyWithInteger):
            return (self.key == other.key and self.integer == other.integer)
        return (self.key == other)

    def __repr__(self):
        return self.key

    def __str__(self):
        return str(self.key)


if __name__ == "__main__":
    c = RawConfigParser(dict_type=OrderedMultiDict)
    c.read("pymod-extensions.cfg")
    print()
    print(c._sections)
    print()
    print(c._sections.items())
    print()
    print(c._sections["all"])
    print()
    print(c._sections["all"].items())
