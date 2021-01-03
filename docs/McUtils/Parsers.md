# <a id="McUtils.Parsers">McUtils.Parsers</a>
    
Utilities for writing parsers of structured text

### Members:

  - [FileStreamReader](Parsers/FileStreamer/FileStreamReader.md)
  - [FileStreamCheckPoint](Parsers/FileStreamer/FileStreamCheckPoint.md)
  - [FileStreamerTag](Parsers/FileStreamer/FileStreamerTag.md)
  - [FileStreamReaderException](Parsers/FileStreamer/FileStreamReaderException.md)
  - [StringStreamReader](Parsers/FileStreamer/StringStreamReader.md)
  - [RegexPattern](Parsers/RegexPatterns/RegexPattern.md)
  - [Capturing](Parsers/RegexPatterns/Capturing.md)
  - [NonCapturing](Parsers/RegexPatterns/NonCapturing.md)
  - [Optional](Parsers/RegexPatterns/Optional.md)
  - [Alternatives](Parsers/RegexPatterns/Alternatives.md)
  - [Longest](Parsers/RegexPatterns/Longest.md)
  - [Shortest](Parsers/RegexPatterns/Shortest.md)
  - [Repeating](Parsers/RegexPatterns/Repeating.md)
  - [Duplicated](Parsers/RegexPatterns/Duplicated.md)
  - [PatternClass](Parsers/RegexPatterns/PatternClass.md)
  - [Parenthesized](Parsers/RegexPatterns/Parenthesized.md)
  - [Named](Parsers/RegexPatterns/Named.md)
  - [Any](Parsers/RegexPatterns/Any.md)
  - [Sign](Parsers/RegexPatterns/Sign.md)
  - [Number](Parsers/RegexPatterns/Number.md)
  - [Integer](Parsers/RegexPatterns/Integer.md)
  - [PositiveInteger](Parsers/RegexPatterns/PositiveInteger.md)
  - [ASCIILetter](Parsers/RegexPatterns/ASCIILetter.md)
  - [AtomName](Parsers/RegexPatterns/AtomName.md)
  - [WhitespaceCharacter](Parsers/RegexPatterns/WhitespaceCharacter.md)
  - [WhitespaceCharacter](Parsers/RegexPatterns/WhitespaceCharacter.md)
  - [Word](Parsers/RegexPatterns/Word.md)
  - [WordCharacter](Parsers/RegexPatterns/WordCharacter.md)
  - [VariableName](Parsers/RegexPatterns/VariableName.md)
  - [CartesianPoint](Parsers/RegexPatterns/CartesianPoint.md)
  - [IntXYZLine](Parsers/RegexPatterns/IntXYZLine.md)
  - [XYZLine](Parsers/RegexPatterns/XYZLine.md)
  - [Empty](Parsers/RegexPatterns/Empty.md)
  - [Newline](Parsers/RegexPatterns/Newline.md)
  - [ZMatPattern](Parsers/RegexPatterns/ZMatPattern.md)
  - [StringParser](Parsers/StringParser/StringParser.md)
  - [StringParserException](Parsers/StringParser/StringParserException.md)
  - [StructuredType](Parsers/StructuredType/StructuredType.md)
  - [StructuredTypeArray](Parsers/StructuredType/StructuredTypeArray.md)
  - [DisappearingTypeClass](Parsers/StructuredType/DisappearingTypeClass.md)
  - [StringParser](Parsers/StringParser/StringParser.md)

### Examples:



```python

from Peeves.TestUtils import *
from unittest import TestCase
from McUtils.Parsers import *
import sys, os, numpy as np

class ParserTests(TestCase):

    @validationTest
    def test_RegexGroups(self):
        # tests whether we capture subgroups or not (by default _not_)

        test_str = "1 2 3 4 a b c d "
        pattern = RegexPattern(
            (
                Capturing(
                    Repeating(
                        Capturing(Repeating(PositiveInteger, 2, 2, suffix=Optional(Whitespace)))
                    )
                ),
                Repeating(Capturing(ASCIILetter), suffix=Whitespace)
            )
        )
        self.assertEquals(len(pattern.search(test_str).groups()), 2)

    @validationTest
    def test_OptScan(self):

        eigsPattern = RegexPattern(
            (
                "Eigenvalues --",
                Repeating(Capturing(Number), suffix=Optional(Whitespace))
            ),
            joiner=Whitespace
        )

        coordsPattern = RegexPattern(
            (
                Capturing(VariableName),
                Repeating(Capturing(Number), suffix=Optional(Whitespace))
            ),
            prefix=Whitespace,
            joiner=Whitespace
        )

        full_pattern = RegexPattern(
            (
                Named(eigsPattern,
                      "Eigenvalues"
                      #parser=lambda t: np.array(Number.findall(t), 'float')
                      ),
                Named(Repeating(coordsPattern, suffix=Optional(Newline)), "Coordinates")
            ),
            joiner=Newline
        )

        with open(TestManager.test_data('scan_params_test.txt')) as test:
            test_str = test.read()

        parser = StringParser(full_pattern)
        parse_res = parser.parse_all(test_str)
        parse_single = parser.parse(test_str)
        parse_its = list(parser.parse_iter(test_str))

        self.assertEquals(parse_res.shape, [(4, 5), [(4, 32), (4, 32, 5)]])
        self.assertIsInstance(parse_res["Coordinates"][1].array, np.ndarray)
        self.assertEquals(int(parse_res["Coordinates"][1, 0].sum()), 3230)

        # print(parse_single["Coordinates"], file = sys.stderr)

    @validationTest
    def test_XYZ(self):

        with open(TestManager.test_data('test_100.xyz')) as test:
            test_str = test.read()

        # print(
        #     "\n".join(test_str.splitlines()[:15]),
        #     "\n",
        #     XYZParser.regex.search(test_str),
        #     file=sys.stderr
        # )

        res = XYZParser.parse_all(
            test_str
        )
        # print(
        #     res["Atoms"],
        #     file=sys.stderr
        # )

        atom_coords = res["Atoms"].array[1].array
        self.assertIsInstance(atom_coords, np.ndarray)
        self.assertEquals(atom_coords.shape, (100, 13, 3))

    @validationTest
    def test_BasicParse(self):
        regex = RegexPattern(
            (
                Named(PositiveInteger, "NumAtoms"),
                Named(
                    Repeating(Any, min = None), "Comment", dtype=str
                ),
                Named(
                    Repeating(
                        Capturing(
                            Repeating(Capturing(Number), 3, 3, prefix = Whitespace, suffix = Optional(Whitespace)),
                            handler= StringParser.array_handler(shape = (None, 3))
                        ),
                        suffix = Optional(Newline)
                    ),
                    "Atoms"
                )
            ),
            "XYZ",
            joiner=Newline
        )

        with open(TestManager.test_data('coord_parse.txt')) as test:
            test_str = test.read()

        res = StringParser(regex).parse(test_str)

        comment_string = res["Comment"].array[0]
        self.assertTrue('comment' in comment_string)
        self.assertEquals(res['Atoms'].array.shape, (4, 3))

        # print(
        #     # regex.dtype,
        #     "",
        #     res,
        #     print(repr(str(regex))),
        #     repr(regex.search(test_str).group("NumAtoms")),
        #     res["NumAtoms"].array,
        #     res['Atoms'].array,
        #     file = sys.stderr,
        #     sep="\n",
        #     end="\n"
        # )
```