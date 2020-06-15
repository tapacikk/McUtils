"""
Implements an importer for Gaussian output formats
"""

import numpy as np, re, math, io
from .GaussianLogComponents import GaussianLogComponents, GaussianLogDefaults, GaussianLogOrdering
from .GaussianFChkComponents import FormattedCheckpointComponents, FormattedCheckpointCommonNames
from ..Parsers import FileStreamReader, FileStreamCheckPoint, FileStreamReaderException

__all__ = ["GaussianFChkReader", "GaussianLogReader"]

########################################################################################################################
#
#                                           GaussianLogReader
#
class GaussianLogReader(FileStreamReader):
    """Implements a stream based reader for a Gaussian .log file... a bit messy

    """

    registered_components = GaussianLogComponents
    default_keys = GaussianLogDefaults
    default_ordering = GaussianLogOrdering

    def parse(self, keys = None, num = None):
        """The main function we'll actually use. Parses bits out of a .log file.

        :param keys: the keys we'd like to read from the log file
        :type keys: str or list(str)
        :param num: for keys with multiple entries, the number of entries to pull
        :type num: int or None
        :return:
        :rtype:
        """
        if keys is None:
            keys = self.default_keys
        # important for ensuring correctness of what we pull
        if isinstance(keys, str):
            keys = (keys,)
        keys = sorted(keys, key = lambda k:self.default_ordering[k] if k in self.default_ordering else 0)
        res = { k:self.parse_key_block(**self.registered_components[k], num=num) for k in keys }
        return res

########################################################################################################################
#
#                                           GaussianFChkReader
#
class GaussianFChkReaderException(FileStreamReaderException):
    pass

class GaussianFChkReader(FileStreamReader):
    """Implements a stream based reader for a Gaussian .fchk file. Pretty generall I think. Should be robust-ish.
    One place to change things up is convenient parsers for specific commonly pulled parts of the fchk

    """

    registered_components = FormattedCheckpointComponents
    common_names = {to_:from_ for from_, to_ in FormattedCheckpointCommonNames.items()}
    to_common_name = FormattedCheckpointCommonNames
    def read_header(self):
        """Reads the header and skips the stream to where we want to be

        :return: the header
        :rtype: str
        """
        return self.get_tagged_block(None, "Number of atoms")

    fchk_re_pattern = r"^(.+?)\s+(I|R|C|H)\s+(N=)?\s+(.+)\s+" # matches name, type, num (if there), and val
    fchk_re = re.compile(fchk_re_pattern)
    def get_next_block_params(self):
        """Pulls the tag of the next block, the type, the number of bytes it'll be,
        and if it's a single-line block it'll also spit back the block itself

        :return:
        :rtype: dict
        """
        with FileStreamCheckPoint(self):
            tag_line = self.readline()
            if tag_line is b'':
                return None
            tag_line = tag_line.decode(self._encoding)
            match = re.match(self.fchk_re, tag_line)
            if match is None:
                raise GaussianFChkReaderException("{}.{}: line '{}' couldn't be read as a tag line".format(
                    type(self).__name__,
                    "get_next_block_params",
                    tag_line
                ))
            jump = self.tell()
        self.seek(jump)

        name, btype, numQ, val = gg = match.groups()
        # print(gg)
        if numQ:
            byte_count = 0
            shits = int(val)
            # hard coded these block formats since they're documented by Gaussian and thus unlikely to change
            if btype == "I":
                byte_per_shit = 12
                shits_per_line = 6
                btype = int
            elif btype == "R":
                byte_per_shit = 16
                shits_per_line = 5
                btype = float
            elif btype == "C":
                byte_per_shit = 12
                shits_per_line = 5
                btype = str
            elif btype == "L":
                byte_per_shit = 12
                shits_per_line = 5
                btype = bool
            byte_count = shits * byte_per_shit + math.ceil( shits / shits_per_line ) # each newline needs a byte
        else:
            byte_count = None
            if btype == "I":
                val = int(val)
            elif btype == "R":
                val = float(val)
            elif btype == "L":
                val = bool(int(val))

        return {
            "name": name,
            "dtype": btype,
            "byte_count": byte_count,
            "value": val
        }

    def get_block(self, name = None, dtype = None, byte_count = None, value = None):
        """Pulls the next block by first pulling the block tag

        :return:
        :rtype:
        """

        if byte_count is not None:

            block_str = self.read(byte_count)
            if dtype in {int, float}:
                block_str = io.StringIO(block_str.decode(self._encoding).replace("\n", "")) # flatten it out
                value = np.loadtxt(block_str)
                if dtype == int:
                    value = value.astype(np.int64)
            else:
                value = block_str

        try:
            parser = self.registered_components[name]
            value = parser(value)
        except KeyError:
            pass

        return value

    def skip_block(self, name = None, dtype = None, byte_count = None, value = None):
        """Skips the next block

        :return:
        :rtype:
        """

        if byte_count is not None:
            self.seek(self.tell() + byte_count)

    def parse(self, keys = None):
        if keys is None:
            keys_to_go = None
        else:
            if isinstance(keys, str):
                keys = (keys,)
            keys_original = set(keys)
            keys_to_go = { (self.common_names[k] if k in self.common_names else k) for k in keys }

        parse_results = {}
        header = self.read_header()

        parse_results['Header'] = header
        if keys_to_go is None:
            while True: # I'll just break once I've exhausted everything
                next_block = self.get_next_block_params()
                if next_block is None:
                    break
                tag = next_block["name"]
                parse_results[tag] = self.get_block(**next_block)
        else:
            while len(keys_to_go)>0:
                next_block = self.get_next_block_params()
                if next_block is None:
                    raise GaussianFChkReaderException("{}.{}: couldn't find keys {}".format(
                        type(self).__name__,
                        "parse",
                        keys_to_go
                        )
                    )
                tag = next_block["name"]
                if tag in keys_to_go:
                    keys_to_go.remove(tag)
                    if tag not in keys_original:
                        tag = self.to_common_name[tag]
                    parse_results[tag] = self.get_block(**next_block)
                else:
                    self.skip_block(**next_block)

        return parse_results


