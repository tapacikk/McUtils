
from Peeves.TestUtils import *
from unittest import TestCase
from McUtils.Jupyter import *
import numpy as np

class JupyterTests(TestCase):

    @validationTest
    def test_HTML(self):
        Div = HTML.Div
        Bootstrap.Panel(
            Bootstrap.Grid(np.random.rand(5, 5).round(3).tolist()),
            header='Test Panel',
            variant='primary'
        ).tostring()

    @debugTest
    def test_Styles(self):
        CSS.parse("""
a {
  text-variant:none;
}
        """)[0].tostring()
