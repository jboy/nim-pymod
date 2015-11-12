# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import sys
if sys.version_info.major >= 3:
    from configparser import RawConfigParser
else:
    from ConfigParser import RawConfigParser

from .OrderedMultiDict import OrderedMultiDict


class UsefulConfigParser(object):
    """A config parser that sucks less than those in module `ConfigParser`."""

    def __init__(self, filenames_to_try=[]):

        # FUN FACT:  In Python 3.2, they spontaneously changed the behaviour of
        # RawConfigParser so that it no longer considers ';' a comment delimiter
        # for inline comments.
        #
        # Compare:
        #   "Configuration files may include comments, prefixed by specific
        #   characters (# and ;). Comments may appear on their own in an otherwise
        #   empty line, or may be entered in lines holding values or section names.
        #   In the latter case, they need to be preceded by a whitespace character
        #   to be recognized as a comment. (For backwards compatibility, only ;
        #   starts an inline comment, while # does not.)"
        #  -- https://docs.python.org/2/library/configparser.html
        # vs:
        #   "Comment prefixes are strings that indicate the start of a valid comment
        #   within a config file. comment_prefixes are used only on otherwise empty
        #   lines (optionally indented) whereas inline_comment_prefixes can be used
        #   after every valid value (e.g. section names, options and empty lines as
        #   well). By default inline comments are disabled and '#' and ';' are used
        #   as prefixes for whole line comments.
        #   Changed in version 3.2: In previous versions of configparser behaviour
        #   matched comment_prefixes=('#',';') and inline_comment_prefixes=(';',)."
        #  -- https://docs.python.org/3/library/configparser.html#customizing-parser-behaviour
        #
        # Grrr...
        if sys.version_info.major >= 3:
            self._cp = RawConfigParser(dict_type=OrderedMultiDict, inline_comment_prefixes=(';',))
        else:
            self._cp = RawConfigParser(dict_type=OrderedMultiDict)

        if isinstance(filenames_to_try, str):
            filenames_to_try = [filenames_to_try]
        self._filenames_to_try = filenames_to_try[:]

    def read(self, filenames_to_try=[]):
        if isinstance(filenames_to_try, str):
            filenames_to_try = [filenames_to_try]
        self._filenames_to_try.extend(filenames_to_try)
        return self._cp.read(self._filenames_to_try)

    def sections(self):
        return self._cp.sections()

    def options(self, section_name):
        ## The client code doesn't need to check in advance that the requested
        ## section name is present in the config; this function will check
        ## this automatically, so no exception is raised by RawConfigParser.

        ## Check that `section_name` is present in the config.
        ## Otherwise, RawConfigParser will raise ConfigParser.NoSectionError.
        if not self._cp.has_section(section_name):
            return []
        return self._cp.options(section_name)

    def get(self, section_name, option_name, do_optionxform=True):
        if do_optionxform:
            # https://docs.python.org/2/library/configparser.html#ConfigParser.RawConfigParser.optionxform
            option_name = self._cp.optionxform(option_name)

        if section_name is None:
            return self._get_optval_in_sections(self.sections(), option_name)
        elif isinstance(section_name, str):
            return self._get_optval_in_sections([section_name], option_name)
        else:
            return self._get_optval_in_sections(section_name, option_name)

    def _get_optval_in_sections(self, section_names, option_name):
        ## The client code doesn't need to check in advance that the requested
        ## section name(s) are present in the config; this function will check
        ## this automatically, so no exception is raised by RawConfigParser.
        optvals = []
        for section_name in section_names:
            ## Check that `section_name` is present in the config.
            ## Otherwise, RawConfigParser will raise ConfigParser.NoSectionError.
            if not self._cp.has_section(section_name):
                continue

            optvals.extend([optval
                    for optname, optval in self._cp.items(section_name)
                    if optname == option_name])
        return optvals

    def getboolean(self, section_name, option_name, do_optionxform=True):
        # https://docs.python.org/2/library/configparser.html#ConfigParser.RawConfigParser.getboolean
        return [self._coerce_to_boolean(optval)
                for optval in self.get(section_name, option_name, do_optionxform)]

    _boolean_states = {'1': True, 'yes': True, 'true': True, 'on': True,
                       '0': False, 'no': False, 'false': False, 'off': False}

    def _coerce_to_boolean(self, optval_str):
        # 'The accepted values for the option are "1", "yes", "true", and "on",
        # which cause this method to return True, and "0", "no", "false", and
        # "off", which cause it to return False. These string values are checked
        # in a case-insensitive manner. Any other value will cause it to raise
        # ValueError.'
        # https://docs.python.org/2/library/configparser.html#ConfigParser.RawConfigParser.getboolean
        ovs_lower = optval_str.lower()
        if ovs_lower not in self._boolean_states:
            raise ValueError("Not a boolean: %s" % optval_str)
        return self._boolean_states[ovs_lower]

