# <a id="McUtils.Plots">McUtils.Plots</a>
    
A plotting framework that builds off of `matplotlib`, but potentially could use a different backend.
The design is intended to mirror the `Graphics` framework in Mathematica and where possible option
names have been chosen to be the same.
Difficulties with `matplotlib` prevent a perfect mirror but the design is consistent.
There are a few primary divisions:
1. `Graphics`/`Graphics3D`/`GraphicsGrid` provide basic access to `matplotlib.figure` and `matplotlib.axes`
    they also hold a `GraphicsPropertyManager`/`GraphicsPropertyManager3D` that manages all properties
    (`image_size`, `axes_label`, `ticks_style`, etc.).
    The full lists can be found on the relevant reference pages and are bound as `properties` on the
    `Graphics`/`Graphics3D` instances.
2. `Plot/Plot3D` and everything in the `Plots` subpackage provide concrete instances of common plots
    with nice names/consistent with Mathematica for discoverability but primarily fall back onto
    `matplotlib` built-in methods and then allow for restyling/data reuse, etc.
3. `Primitives` provide direct access to the shapes that are actually plotted on screen (i.e. `matplotlib.Patch` objects)
    in a convenient way to add on to existing plots
4. `Styling` provides access to theme management/construction

Image/animation support and other back end support for 3D graphics (`VTK`) are provided at the experimental level.

### Members

  - [GraphicsBase](Plots/Graphics/GraphicsBase.md)
  - [Graphics](Plots/Graphics/Graphics.md)
  - [Graphics3D](Plots/Graphics/Graphics3D.md)
  - [GraphicsGrid](Plots/Graphics/GraphicsGrid.md)
  - [Plot](Plots/Plots/Plot.md)
  - [ScatterPlot](Plots/Plots/ScatterPlot.md)
  - [ErrorBarPlot](Plots/Plots/ErrorBarPlot.md)
  - [ListErrorBarPlot](Plots/Plots/ListErrorBarPlot.md)
  - [StickPlot](Plots/Plots/StickPlot.md)
  - [ListStickPlot](Plots/Plots/ListStickPlot.md)
  - [TriPlot](Plots/Plots/TriPlot.md)
  - [ListTriPlot](Plots/Plots/ListTriPlot.md)
  - [DataPlot](Plots/Plots/DataPlot.md)
  - [HistogramPlot](Plots/Plots/HistogramPlot.md)
  - [HistogramPlot2D](Plots/Plots/HistogramPlot2D.md)
  - [VerticalLinePlot](Plots/Plots/VerticalLinePlot.md)
  - [ArrayPlot](Plots/Plots/ArrayPlot.md)
  - [TensorPlot](Plots/Plots/TensorPlot.md)
  - [Plot2D](Plots/Plots/Plot2D.md)
  - [ListPlot2D](Plots/Plots/ListPlot2D.md)
  - [ContourPlot](Plots/Plots/ContourPlot.md)
  - [DensityPlot](Plots/Plots/DensityPlot.md)
  - [ListContourPlot](Plots/Plots/ListContourPlot.md)
  - [ListDensityPlot](Plots/Plots/ListDensityPlot.md)
  - [ListTriContourPlot](Plots/Plots/ListTriContourPlot.md)
  - [ListTriDensityPlot](Plots/Plots/ListTriDensityPlot.md)
  - [ListTriPlot3D](Plots/Plots/ListTriPlot3D.md)
  - [Plot3D](Plots/Plots/Plot3D.md)
  - [ListPlot3D](Plots/Plots/ListPlot3D.md)
  - [ScatterPlot3D](Plots/Plots/ScatterPlot3D.md)
  - [WireframePlot3D](Plots/Plots/WireframePlot3D.md)
  - [ContourPlot3D](Plots/Plots/ContourPlot3D.md)
  - [GraphicsPrimitive](Plots/Primitives/GraphicsPrimitive.md)
  - [Sphere](Plots/Primitives/Sphere.md)
  - [Cylinder](Plots/Primitives/Cylinder.md)
  - [Disk](Plots/Primitives/Disk.md)
  - [Line](Plots/Primitives/Line.md)
  - [EventHandler](Plots/Interactive/EventHandler.md)
  - [Animator](Plots/Interactive/Animator.md)
  - [Styled](Plots/Styling/Styled.md)
  - [ThemeManager](Plots/Styling/ThemeManager.md)
  - [Image](Plots/Image/Image.md)
  - [GraphicsPropertyManager](Plots/Properties/GraphicsPropertyManager.md)
  - [GraphicsPropertyManager3D](Plots/Properties/GraphicsPropertyManager3D.md)

### Examples



### Unit Tests

```python

from Peeves.TestUtils import *
from unittest import TestCase
from McUtils.Plots import *
import sys, os, numpy as np

class PlotsTests(TestCase):

    @classmethod
    def tearDownClass(cls):
        import matplotlib.pyplot as plt
        # plt.show()

    def result_file(self, fname):
        if not os.path.isdir(os.path.join(TestManager.test_dir, "test_results")):
            os.mkdir(os.path.join(TestManager.test_dir, "test_results"))
        return os.path.join(TestManager.test_dir, "test_results", fname)

    @validationTest
    def test_Plot(self):
        grid = np.linspace(0, 2*np.pi, 100)
        plot = Plot(grid, np.sin(grid))

        plot.savefig(self.result_file("test_Plot.png"))
        plot.close()
        # plot.show()

    @validationTest
    def test_Plot3D(self):
        import matplotlib.cm as colormaps

        f = lambda pt: np.sin(pt[0]) + np.cos(pt[1])
        plot = Plot3D(f, np.arange(0, 2 * np.pi, .1), np.arange(0, 2 * np.pi, .1),
                      plot_style={
                          "cmap": colormaps.get_cmap('viridis')
                      },
                      axes_labels=['dogs', 'cats',
                                   Styled('rats', color='red')
                                   ],
                      plot_label='my super cool 3D plot',
                      plot_range=[(-5, 5)] * 3,
                      plot_legend='i lik turtle',
                      colorbar=True
                      )
        plot.savefig(self.result_file("test_Plot3D.png"))
        plot.close()

    @validationTest
    def test_GraphicsGrid(self):

        main = GraphicsGrid(ncols=3, nrows=1)
        grid = np.linspace(0, 2 * np.pi, 100)
        grid_2D = np.meshgrid(grid, grid)
        main[0, 0] = ContourPlot(grid_2D[1], grid_2D[0], np.sin(grid_2D[0]), figure=main[0, 0])
        main[0, 1] = ContourPlot(grid_2D[1], grid_2D[0], np.sin(grid_2D[0]) * np.cos(grid_2D[1]), figure=main[0, 1])
        main[0, 2] = ContourPlot(grid_2D[1], grid_2D[0], np.cos(grid_2D[1]), figure=main[0, 2])
        # main.show()

        main.savefig(self.result_file("test_GraphicsGrid.png"))
        main.close()

    @validationTest
    def test_PlotStyling(self):
        grid = np.linspace(0, 2 * np.pi, 100)
        # file = '~/Desktop/y.png'
        plot = Plot(grid, np.sin(grid),
                    aspect_ratio=1.3,
                    theme='dark_background',
                    ticks_style={'color':'red', 'labelcolor':'red'},
                    plot_label='bleh',
                    padding=((30, 0), (20, 20))
                    )
        # plot.savefig(file)
        # plot = Image.from_file(file)
        # plot.show()
        plot.savefig(self.result_file("test_PlotStyling.png"))
        plot.close()

    @debugTest
    def test_PlotGridStyling(self):
        main = GraphicsGrid(ncols=3, nrows=1, theme='Solarize_Light2', figure_label='my beuatufil triptych',
                            padding=((35, 60), (35, 40)))
        grid = np.linspace(0, 2 * np.pi, 100)
        grid_2D = np.meshgrid(grid, grid)
        x = grid_2D[1]; y = grid_2D[0]
        main[0, 0] = ContourPlot(x, y, np.sin(y), plot_label='$sin(x)$',
                                 axes_labels=[None, "cats (cc)"],
                                 figure=main[0, 0]
                                 )
        main[0, 1] = ContourPlot(x, y, np.sin(x) * np.cos(y),
                                 plot_label='$sin(x)cos(y)$',
                                 axes_labels=[Styled("dogs (arb.)", {'color': 'red'}), None],
                                 figure=main[0, 1])
        main[0, 2] = ContourPlot(x, y, np.cos(y), plot_label='$cos(y)$', figure=main[0, 2])
        main.colorbar = {"graphics": main[0, 1].graphics}

        # main.show()

        main.savefig(self.result_file("test_PlotGridStyling.png"))
        main.close()

    @debugTest
    def test_Scatter(self):
        pts = np.random.rand(50, 2)
        plot = ScatterPlot(*pts.T,
                           aspect_ratio=2,
                           image_size=250
                           )
        # plot.show()
        plot.savefig(self.result_file("test_Scatter.pdf"), format='pdf')
        plot.close()

    @validationTest
    def test_ListContourPlot(self):
        pts = np.pi*np.random.rand(150, 2)
        sins = np.sin(pts[:, 0])
        coses = np.cos(pts[:, 1])
        ptss = np.concatenate((pts, np.reshape(sins*coses, sins.shape + (1,))), axis=1)
        plot = ListContourPlot(ptss)

        plot.savefig(self.result_file("test_ListContourPlot.png"))
        plot.close()

    @validationTest
    def test_ListTriPlot3D(self):
        pts = np.pi*np.random.rand(150, 2)
        sins = np.sin(pts[:, 0])
        coses = np.cos(pts[:, 1])
        ptss = np.concatenate((pts, np.reshape(sins*coses, sins.shape + (1,))), axis=1)
        plot = ListTriPlot3D(ptss)

        plot.savefig(self.result_file("test_ListTriPlot3D.png"))
        plot.close()

    @validationTest
    def test_ListTriDensityPlot(self):
        pts = np.pi*np.random.rand(150, 2)
        sins = np.sin(pts[:, 0])
        coses = np.cos(pts[:, 1])
        ptss = np.concatenate((pts, np.reshape(sins*coses, sins.shape + (1,))), axis=1)
        plot = ListTriDensityPlot(ptss)

        plot.savefig(self.result_file("test_ListTriDensityPlot.png"))
        plot.close()

    @validationTest
    def test_ListTriContourPlot(self):
        pts = np.pi*np.random.rand(150, 2)
        sins = np.sin(pts[:, 0])
        coses = np.cos(pts[:, 1])
        ptss = np.concatenate((pts, np.reshape(sins*coses, sins.shape + (1,))), axis=1)
        plot = ListTriContourPlot(ptss)
        plot.add_colorbar()

        plot.savefig(self.result_file("test_ListTriContourPlot.png"))
        plot.close()

    @inactiveTest
    def test_Animation(self):
        "Currently broken"
        def get_data(*args):
            pts = np.pi*np.random.normal(scale = .25, size=(10550, 2))
            sins = np.sin(pts[:, 0])
            coses = np.cos(pts[:, 1])
            ptss = np.concatenate((pts, np.reshape(sins*coses, sins.shape + (1,))), axis=1)
            return (ptss, )
        plot = ListTriContourPlot(*get_data(),
                                  animate = get_data,
                                  plot_range = [
                                      [-np.pi, np.pi],
                                      [-np.pi, np.pi]
                                  ]
                                  )

        plot.show()

        plot.savefig(self.result_file("test_ListTriContourPlot.gif"))
        plot.close()

    @inactiveTest
    def test_VTK(self):
        plot = Graphics3D(backend="VTK", image_size=[1500, 500])
        Sphere().plot(plot)
        # plot.show()

    # @validationTest
    # def test_Plot3D_adaptive(self):
    #     f = lambda pt: np.sin(pt[0]) + np.cos(pt[1])
    #     plot = Plot3D(f, [0, 2*np.pi], [0, 2*np.pi])
    #     plot.show()

    @validationTest
    def test_PlotDelayed(self):
        p = Plot(background = 'black')
        for i, c in enumerate(('red', 'white', 'blue')):
            p.plot(np.sin, [-2 + 4/3*i, -2 + 4/3*(i+1)], color = c)
        # p.show()

        p.savefig(self.result_file("test_PlotDelayed.gif"))
        p.close()

    @validationTest
    def test_Plot3DDelayed(self):
        p = Plot3D(background = 'black')
        for i, c in enumerate(('red', 'white', 'blue')):
            p.plot(
                lambda g: (
                    np.sin(g.T[0]) + np.cos(g.T[1])
                ),
                [-2 + 4/3*i, -2 + 4/3*(i+1)],
                [-2 + 4/3*i, -2 + 4/3*(i+1)],
                color = c)
        # p.show()

        p.savefig(self.result_file("test_Plot3DDelayed.gif"))
        p.close()
```

___

[Edit Examples](https://github.com/McCoyGroup/McUtils/edit/edit/ci/examples/ci/docs/McUtils/Plots.md) or 
[Create New Examples](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/examples/ci/docs/McUtils/Plots.md) <br/>
[Edit Template](https://github.com/McCoyGroup/McUtils/edit/edit/ci/docs/ci/docs/McUtils/Plots.md) or 
[Create New Template](https://github.com/McCoyGroup/McUtils/new/edit/?filename=ci/docs/templates/ci/docs/McUtils/Plots.md) <br/>
[Edit Docstrings](https://github.com/McCoyGroup/McUtils/edit/edit/McUtils/Plots/__init__.py?message=Update%20Docs)