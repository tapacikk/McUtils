"""
A plotting framework that builds off of matplotlib, but potentially could use a different backend
"""

from .Graphics import Graphics, Graphics3D, GraphicsGrid
from .Plots import Plot, ScatterPlot, ErrorBarPlot, ListErrorBarPlot, StickPlot, ListStickPlot, TriPlot, ListTriPlot, \
    DataPlot, HistogramPlot, HistogramPlot2D, VerticalLinePlot, ArrayPlot, TensorPlot, \
    Plot2D, ListPlot2D, ContourPlot, ListContourPlot, ListDensityPlot, ListTriContourPlot, ListTriDensityPlot, ListTriPlot3D,\
    Plot3D, ListPlot3D, ScatterPlot3D, WireframePlot3D, ContourPlot3D
from .Primitives import GraphicsPrimitive, Sphere, Cylinder, Disk, Line