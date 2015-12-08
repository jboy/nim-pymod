#!/usr/bin/env python
# Regenerate the table of contents in the Pymod README.
#
# Gets the latest version of the Pymod README (rendered using Github Markdown)
# from the webpage for the "nim-pymod" repo on the Github website.
# Extracts the section headings & Github Markdown-generated href fragments from
# the fetched HTML, then prints them to stdout as a list of links formatted in
# Markdown.  You then copy-paste the output back into "README.md".
#
# Usage:
#  python regenerate_toc.py

from __future__ import print_function

PYMOD_URL = "https://github.com/jboy/nim-pymod"

import re
import sys

import requests


def main():
    r = requests.get(PYMOD_URL)
    if r.status_code != 200:
        die("unable to GET (status code == %d) from URL: %s" % (r.status_code, PYMOD_URL))
    process_response_text(r.text)


def process_response_text(text):
    toc_entry_num = 0
    for i, line in enumerate(text.split("\n")):
        if "id=\"user-content-" in line:
            toc_entry_num = process_section_heading(i+1, line, toc_entry_num)


_HEADINGS_TO_IGNORE = ["Pymod", "Table of contents"]

_LINE_PATTERN = "^.* href=\"(?P<href>#[^\"]+)\" .*</a>(?P<heading>[^<]+)</h.*$"
_LINE_REGEX = re.compile(_LINE_PATTERN)

def process_section_heading(line_num, line, toc_entry_num):
    m = _LINE_REGEX.match(line)
    if not m:
        die("line %d failed regex match:\n %s" % (line_num, line))
    href = m.group("href")
    heading = m.group("heading")
    #print((href, heading))
    heading = heading.replace("&amp;", "&")

    if heading not in _HEADINGS_TO_IGNORE:
        toc_entry_num += 1
        print("%d. [%s](%s)" % (toc_entry_num, heading, href))
    return toc_entry_num


def die(msg):
    print("%s: %s\nAborting." % (sys.argv[0], msg), file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
