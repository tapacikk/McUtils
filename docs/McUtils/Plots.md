# <a id="McUtils.Plots">McUtils.Plots</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Plots/__init__.py#L1)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/__init__.py#L1?message=Update%20Docs)]
</div>
    
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
<div class="container alert alert-secondary bg-light">
  <div class="row">
   <div class="col" markdown="1">
[GraphicsBase](Plots/Graphics/GraphicsBase.md)   
</div>
   <div class="col" markdown="1">
[Graphics](Plots/Graphics/Graphics.md)   
</div>
   <div class="col" markdown="1">
[Graphics3D](Plots/Graphics/Graphics3D.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[GraphicsGrid](Plots/Graphics/GraphicsGrid.md)   
</div>
   <div class="col" markdown="1">
[Plot](Plots/Plots/Plot.md)   
</div>
   <div class="col" markdown="1">
[DataPlot](Plots/Plots/DataPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[ArrayPlot](Plots/Plots/ArrayPlot.md)   
</div>
   <div class="col" markdown="1">
[TensorPlot](Plots/Plots/TensorPlot.md)   
</div>
   <div class="col" markdown="1">
[Plot2D](Plots/Plots/Plot2D.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[ListPlot2D](Plots/Plots/ListPlot2D.md)   
</div>
   <div class="col" markdown="1">
[Plot3D](Plots/Plots/Plot3D.md)   
</div>
   <div class="col" markdown="1">
[ListPlot3D](Plots/Plots/ListPlot3D.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[FilledPlot](Plots/Plots/FilledPlot.md)   
</div>
   <div class="col" markdown="1">
[ScatterPlot](Plots/Plots/ScatterPlot.md)   
</div>
   <div class="col" markdown="1">
[ErrorBarPlot](Plots/Plots/ErrorBarPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[StickPlot](Plots/Plots/StickPlot.md)   
</div>
   <div class="col" markdown="1">
[DatePlot](Plots/Plots/DatePlot.md)   
</div>
   <div class="col" markdown="1">
[StepPlot](Plots/Plots/StepPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[LogLogPlot](Plots/Plots/LogLogPlot.md)   
</div>
   <div class="col" markdown="1">
[SemiLogXPlot](Plots/Plots/SemiLogXPlot.md)   
</div>
   <div class="col" markdown="1">
[SemilogYPlot](Plots/Plots/SemilogYPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[HorizontalFilledPlot](Plots/Plots/HorizontalFilledPlot.md)   
</div>
   <div class="col" markdown="1">
[BarPlot](Plots/Plots/BarPlot.md)   
</div>
   <div class="col" markdown="1">
[HorizontalBarPlot](Plots/Plots/HorizontalBarPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[EventPlot](Plots/Plots/EventPlot.md)   
</div>
   <div class="col" markdown="1">
[PiePlot](Plots/Plots/PiePlot.md)   
</div>
   <div class="col" markdown="1">
[StackPlot](Plots/Plots/StackPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[BrokenHorizontalBarPlot](Plots/Plots/BrokenHorizontalBarPlot.md)   
</div>
   <div class="col" markdown="1">
[HorizontalLinePlot](Plots/Plots/HorizontalLinePlot.md)   
</div>
   <div class="col" markdown="1">
[PolygonPlot](Plots/Plots/PolygonPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[AxisHorizontalLinePlot](Plots/Plots/AxisHorizontalLinePlot.md)   
</div>
   <div class="col" markdown="1">
[AxisHorizontalSpanPlot](Plots/Plots/AxisHorizontalSpanPlot.md)   
</div>
   <div class="col" markdown="1">
[AxisVerticalLinePlot](Plots/Plots/AxisVerticalLinePlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[AxisVeticalSpanPlot](Plots/Plots/AxisVeticalSpanPlot.md)   
</div>
   <div class="col" markdown="1">
[AxisLinePlot](Plots/Plots/AxisLinePlot.md)   
</div>
   <div class="col" markdown="1">
[StairsPlot](Plots/Plots/StairsPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[HistogramPlot](Plots/Plots/HistogramPlot.md)   
</div>
   <div class="col" markdown="1">
[HistogramPlot2D](Plots/Plots/HistogramPlot2D.md)   
</div>
   <div class="col" markdown="1">
[SpectrogramPlot](Plots/Plots/SpectrogramPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[AutocorrelationPlot](Plots/Plots/AutocorrelationPlot.md)   
</div>
   <div class="col" markdown="1">
[AngleSpectrumPlot](Plots/Plots/AngleSpectrumPlot.md)   
</div>
   <div class="col" markdown="1">
[CoherencePlot](Plots/Plots/CoherencePlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[CrossSpectralDensityPlot](Plots/Plots/CrossSpectralDensityPlot.md)   
</div>
   <div class="col" markdown="1">
[MagnitudeSpectrumPlot](Plots/Plots/MagnitudeSpectrumPlot.md)   
</div>
   <div class="col" markdown="1">
[PhaseSpectrumPlot](Plots/Plots/PhaseSpectrumPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[PowerSpectralDensityPlot](Plots/Plots/PowerSpectralDensityPlot.md)   
</div>
   <div class="col" markdown="1">
[CrossCorrelationPlot](Plots/Plots/CrossCorrelationPlot.md)   
</div>
   <div class="col" markdown="1">
[BoxPlot](Plots/Plots/BoxPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[ViolinPlot](Plots/Plots/ViolinPlot.md)   
</div>
   <div class="col" markdown="1">
[BoxAndWhiskerPlot](Plots/Plots/BoxAndWhiskerPlot.md)   
</div>
   <div class="col" markdown="1">
[HexagonalHistogramPlot](Plots/Plots/HexagonalHistogramPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[QuiverPlot](Plots/Plots/QuiverPlot.md)   
</div>
   <div class="col" markdown="1">
[StreamPlot](Plots/Plots/StreamPlot.md)   
</div>
   <div class="col" markdown="1">
[MatrixPlot](Plots/Plots/MatrixPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[SparsityPlot](Plots/Plots/SparsityPlot.md)   
</div>
   <div class="col" markdown="1">
[ContourPlot](Plots/Plots/ContourPlot.md)   
</div>
   <div class="col" markdown="1">
[DensityPlot](Plots/Plots/DensityPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[HeatmapPlot](Plots/Plots/HeatmapPlot.md)   
</div>
   <div class="col" markdown="1">
[TriPlot](Plots/Plots/TriPlot.md)   
</div>
   <div class="col" markdown="1">
[TriDensityPlot](Plots/Plots/TriDensityPlot.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[TriContourLinesPlot](Plots/Plots/TriContourLinesPlot.md)   
</div>
   <div class="col" markdown="1">
[TriContourPlot](Plots/Plots/TriContourPlot.md)   
</div>
   <div class="col" markdown="1">
[ScatterPlot3D](Plots/Plots/ScatterPlot3D.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[WireframePlot3D](Plots/Plots/WireframePlot3D.md)   
</div>
   <div class="col" markdown="1">
[ContourPlot3D](Plots/Plots/ContourPlot3D.md)   
</div>
   <div class="col" markdown="1">
[GraphicsPrimitive](Plots/Primitives/GraphicsPrimitive.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Sphere](Plots/Primitives/Sphere.md)   
</div>
   <div class="col" markdown="1">
[Cylinder](Plots/Primitives/Cylinder.md)   
</div>
   <div class="col" markdown="1">
[Disk](Plots/Primitives/Disk.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Line](Plots/Primitives/Line.md)   
</div>
   <div class="col" markdown="1">
[Text](Plots/Primitives/Text.md)   
</div>
   <div class="col" markdown="1">
[Arrow](Plots/Primitives/Arrow.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Inset](Plots/Primitives/Inset.md)   
</div>
   <div class="col" markdown="1">
[EventHandler](Plots/Interactive/EventHandler.md)   
</div>
   <div class="col" markdown="1">
[Animator](Plots/Interactive/Animator.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Styled](Plots/Styling/Styled.md)   
</div>
   <div class="col" markdown="1">
[ThemeManager](Plots/Styling/ThemeManager.md)   
</div>
   <div class="col" markdown="1">
[PlotLegend](Plots/Styling/PlotLegend.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Image](Plots/Image/Image.md)   
</div>
   <div class="col" markdown="1">
[GraphicsPropertyManager](Plots/Properties/GraphicsPropertyManager.md)   
</div>
   <div class="col" markdown="1">
[GraphicsPropertyManager3D](Plots/Properties/GraphicsPropertyManager3D.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
</div>













---


<div markdown="1" class="text-secondary">
<div class="container">
  <div class="row">
   <div class="col" markdown="1">
**Feedback**   
</div>
   <div class="col" markdown="1">
**Examples**   
</div>
   <div class="col" markdown="1">
**Templates**   
</div>
   <div class="col" markdown="1">
**Documentation**   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[Bug](https://github.com/McCoyGroup/McUtils/issues/new?title=Documentation%20Improvement%20Needed)/[Request](https://github.com/McCoyGroup/McUtils/issues/new?title=Example%20Request)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Plots.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Plots.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Plots.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Plots.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Plots/__init__.py#L1?message=Update%20Docs)   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
   <div class="col" markdown="1">
   
</div>
</div>
</div>
</div>