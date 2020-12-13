from mmap import mmap

__all__ = [
    "FileStreamReader",
    "FileStreamCheckPoint",
    "FileStreamerTag",
    "FileStreamReaderException"
]

########################################################################################################################
#
#                                           FileStreamReader
#
class FileStreamCheckPoint:
    """
    A checkpoint for a file that can be returned to when parsing
    """
    def __init__(self, parent, revert = True):
        self.parent = parent
        self.chk = parent.tell()
        self._revert = revert
    def revert(self):
        self.parent.seek(self.chk)
    def __enter__(self, ):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._revert:
            self.revert()

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
        self._wasopen = None

    def __enter__(self):
        if isinstance(self._file, str):
            self._wasopen = False
            self._fstream = open(self._file, mode=self._mode, **self._kw)
        else:
            self._wasopen = True
            self._fstream = self._file
        self._stream = mmap(self._fstream.fileno(), 0)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stream.close()
        if not self._wasopen:
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

    def _find_tag(self, tag,
                  skip_tag = True,
                  seek = True,
                  direction = 'forward'
                  ):
        """Finds a tag in a file

        :param header: a tag specifying a header to look for + optional follow-up processing/offsets
        :type header: FileStreamerTag
        :return: position of tag
        :rtype: int
        """
        with FileStreamCheckPoint(self, revert = False) as chk:
            enc_tag = tag.encode(self._encoding)
            mode = self._stream.find if direction == 'forward' else self._stream.rfind
            pos = mode(enc_tag)
            if seek and pos >= 0:
                if skip_tag:
                    pos = pos + len(enc_tag) if direction == 'forward' else pos - len(enc_tag)
                self._stream.seek(pos)
            elif pos == -1:
                chk.revert()
        return pos
    def find_tag(self, tag,
                 skip_tag = None,
                 seek = None
                 ):
        """Finds a tag in a file

        :param header: a tag specifying a header to look for + optional follow-up processing/offsets
        :type header: FileStreamerTag
        :return: position of tag
        :rtype: int
        """
        if isinstance(tag, str):
            tags = FileStreamerTag(tag)
        else:
            tags = tag

        pos = -1
        if skip_tag is None:
            skip_tag = tags.skip_tag
        if seek is None:
            seek = tags.seek
        for i, tag in enumerate(tags.tags):
            pos = self._find_tag(tag,
                                 skip_tag = skip_tag,
                                 seek = seek
                                 )
            if pos == -1:
                continue

            follow_ups = tags.skips
            if follow_ups is not None:
                for tag in follow_ups:
                    self.seek(pos + 1)
                    p = self.find_tag(tag)
                    if p > -1:
                        pos = p

            offset = tags.offset
            if offset is not None:
                # why are we using self._stream.tell here...?
                # I won't touch it for now but I feel like it should be pos
                pos = self._stream.tell() + offset
                self._stream.seek(pos)

        return pos

    def get_tagged_block(self, tag_start, tag_end, block_size = 500):
        """Pulls the string between tag_start and tag_end

        :param tag_start:
        :type tag_start: FileStreamerTag or None
        :param tag_end:
        :type tag_end: FileStreamerTag
        :return:
        :rtype:
        """
        if tag_start is not None:
            start = self.find_tag(tag_start)
            if start >= 0:
                with FileStreamCheckPoint(self):
                    end = self.find_tag(tag_end, seek=False)
                if end > 0:
                    return self._stream.read(end-start).decode(self._encoding)
        else:
            start = self.tell()
            with FileStreamCheckPoint(self):
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
            with FileStreamCheckPoint(self):
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

class FileStreamerTag:
    def __init__(self,
                 tag_alternatives = None,
                 follow_ups = None,
                 offset = None,
                 direction = "forward",
                 skip_tag = True,
                 seek = True
                 ):
        if tag_alternatives is None:
            raise FileStreamReaderException("{} needs to be supplied with some set of tag_alternatives to look for".format(
                type(self).__name__
            ))
        self.tags = (tag_alternatives,) if isinstance(tag_alternatives, str) else tag_alternatives
        self.skips = (follow_ups,) if isinstance(follow_ups, (str, FileStreamerTag)) else follow_ups
        self.offset = offset
        self.direction = direction
        self.skip_tag = skip_tag
        self.seek = seek