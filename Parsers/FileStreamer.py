from mmap import mmap
__all__ = [
    "FileStreamReader",
    "FileCheckPoint",
    "FileStreamReaderException"
]

########################################################################################################################
#
#                                           FileStreamReader
#
class FileCheckPoint:
    """
    A checkpoint for a file that can be returned to when parsing
    """
    def __init__(self, parent):
        self._parent = parent
        self._chk = parent.tell()
    def __enter__(self, ):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._parent.seek(self._chk)

class FileStreamReaderException(IOError):
    pass

class FileStreamReader:
    """
    Represents a file from which we'll stream blocks of data by finding tags and parsing what's between them
    """
    def __init__(self, file, mode="r", encoding="utf-8", **kw):
        self._file = file
        self._mode = mode.strip("+b")+"+b"
        self._encoding = encoding
        self._kw = kw
        self._stream = None

    def __enter__(self):
        self._fstream = open(self._file, mode=self._mode, **self._kw)
        self._stream = mmap(self._fstream.fileno(), 0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stream.close()
        self._fstream.close()

    def __iter__(self):
        return iter(self._fstream)

    def __getattr__(self, item):
        return getattr(self._stream, item)
    def readline(self):
        return self._stream.readline()
    def read(self, n=1):
        return self._stream.read(n)
    def seek(self, *args, **kwargs):
        # self._fstream.seek(*args, **kwargs)
        return self._stream.seek(*args, **kwargs)
    def tell(self):
        return self._stream.tell()

    def find_tag(self, tag, skip_tag = True, seek = True):
        """Finds a tag in a file

        :param header: a string specifying a header to look for
        :type header: str
        :return: if header was found
        :rtype: bool
        """
        if isinstance(tag, str):
            tags = (tag,)
        else:
            tags = tag

        pos = -1
        for i, tag in enumerate(tags):
            enc_tag = tag.encode(self._encoding)
            pos = self._stream.find(enc_tag)
            if seek and pos > 0:
                if skip_tag:
                    pos = pos + len(enc_tag)
                self._stream.seek(pos)
            elif pos < 0:
                break
        return pos

    def get_tagged_block(self, tag_start, tag_end, block_size = 500):
        """Pulls the string between tag_start and tag_end

        :param tag_start:
        :type tag_start: str or None
        :param tag_end:
        :type tag_end: str
        :return:
        :rtype:
        """
        if tag_start is not None:
            start = self.find_tag(tag_start)
            if start > 0:
                with FileCheckPoint(self):
                    end = self.find_tag(tag_end, seek=False)
                if end > 0:
                    return self._stream.read(end-start).decode(self._encoding)
        else:
            start = self.tell()
            with FileCheckPoint(self):
                end = self.find_tag(tag_end, seek=False)
            if end > 0:
                return self._stream.read(end-start).decode(self._encoding)

        # implict None return if no block found

    def parse_key_block(self, tag_start=None, tag_end=None, mode="Single", parser = None, parse_mode = "List", num = None, **ignore):
        """Parses a block by starting at tag_start and looking for tag_end and parsing what's in between them

        :param key: registered key pattern to pull from a file
        :type key: str
        :return:
        :rtype:
        """
        if tag_start is None:
            raise FileStreamReaderException("{}.{}: needs a '{}' argument".format(
                type(self).__name__,
                "parse_key_block",
                "tag_start"
            ))
        if tag_end is None:
            raise FileStreamReaderException("{}.{}: needs a '{}' argument".format(
                type(self).__name__,
                "parse_key_block",
                "tag_end"
            ))
        if mode == "List":
            with FileCheckPoint(self):
                # we do this in a checkpointed fashion only for list-type tokens
                # for all other tokens we introduce an ordering to apply when checking
                # does it need to be done like this... probably not?
                # I suppose we could be a significantly more efficient by returning a
                # generator statement in these multi-block cases
                # and then introducing a sorting order across these multi-blocks
                # that tells us which to check first, second, etc.
                # but this is probably good enough
                if isinstance(num, int):
                    blocks = [None]*num
                    if parser is None:
                        parser = lambda a:a

                    i = 0 # protective
                    for i in range(num):
                        block = self.get_tagged_block(tag_start, tag_end)
                        if block is None:
                            break
                        if parse_mode != "List":
                            block = parser(block)
                        blocks[i] = block

                    if parse_mode == "List":
                        blocks = parser(blocks[:i+1])
                else:
                    blocks = []
                    block = self.get_tagged_block(tag_start, tag_end)
                    if parser is None:
                        parser = lambda a:a
                    while block is not None:
                        if parse_mode != "List":
                            block = parser(block)
                        blocks.append(block)
                        block = self.get_tagged_block(tag_start, tag_end)

                    if parse_mode == "List":
                        blocks = parser(blocks)
                return blocks
        else:
            block = self.get_tagged_block(tag_start, tag_end)
            if parser is not None:
                block = parser(block)
            return block