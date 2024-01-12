# <a id="McUtils.Coordinerds">McUtils.Coordinerds</a> 
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Coordinerds/__init__.py#L1)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/__init__.py#L1?message=Update%20Docs)]
</div>
    
The Coordinerds package implements stuff for dealing with coordinates and generalized coordinate systems

It provides a semi-symbolic way to represent a CoordinateSystem and a CoordinateSet that provides coordinates within a
coordinate system. An extensible system for converting between coordinate systems and is provided.

The basic design of the package is set up so that one creates a `CoordinateSet` object, which in turn tracks its `CoordinateSystem`.
A `CoordinateSet` is a subclass of `np.ndarray`, and so any operation that works for a `np.ndarray` will work in turn for `CoordinateSet`.
This provides a large amount flexibility.

The `CoordinateSystem` object handles much of the heavy lifting for a `CoordinateSet`.
Conversions between different systems are implemented by a `CoordinateSystemConverter`.
Chained conversions are not _currently_ supported, but might well become supported in the future.

### Members
<div class="container alert alert-secondary bg-light">
  <div class="row">
   <div class="col" markdown="1">
[CoordinateSystemConverters](Coordinerds/CoordinateSystems/CoordinateSystemConverter/CoordinateSystemConverters.md)   
</div>
   <div class="col" markdown="1">
[CoordinateSystemConverter](Coordinerds/CoordinateSystems/CoordinateSystemConverter/CoordinateSystemConverter.md)   
</div>
   <div class="col" markdown="1">
[SimpleCoordinateSystemConverter](Coordinerds/CoordinateSystems/CoordinateSystemConverter/SimpleCoordinateSystemConverter.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[CartesianCoordinateSystem](Coordinerds/CoordinateSystems/CommonCoordinateSystems/CartesianCoordinateSystem.md)   
</div>
   <div class="col" markdown="1">
[InternalCoordinateSystem](Coordinerds/CoordinateSystems/CommonCoordinateSystems/InternalCoordinateSystem.md)   
</div>
   <div class="col" markdown="1">
[CartesianCoordinateSystem3D](Coordinerds/CoordinateSystems/CommonCoordinateSystems/CartesianCoordinateSystem3D.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[CartesianCoordinates3D](Coordinerds/CoordinateSystems/CommonCoordinateSystems/CartesianCoordinates3D.md)   
</div>
   <div class="col" markdown="1">
[CartesianCoordinates1D](Coordinerds/CoordinateSystems/CommonCoordinateSystems/CartesianCoordinates1D.md)   
</div>
   <div class="col" markdown="1">
[CartesianCoordinates2D](Coordinerds/CoordinateSystems/CommonCoordinateSystems/CartesianCoordinates2D.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[SphericalCoordinateSystem](Coordinerds/CoordinateSystems/CommonCoordinateSystems/SphericalCoordinateSystem.md)   
</div>
   <div class="col" markdown="1">
[SphericalCoordinates](Coordinerds/CoordinateSystems/CommonCoordinateSystems/SphericalCoordinates.md)   
</div>
   <div class="col" markdown="1">
[ZMatrixCoordinateSystem](Coordinerds/CoordinateSystems/CommonCoordinateSystems/ZMatrixCoordinateSystem.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[ZMatrixCoordinates](Coordinerds/CoordinateSystems/CommonCoordinateSystems/ZMatrixCoordinates.md)   
</div>
   <div class="col" markdown="1">
[CoordinateSystem](Coordinerds/CoordinateSystems/CoordinateSystem/CoordinateSystem.md)   
</div>
   <div class="col" markdown="1">
[BaseCoordinateSystem](Coordinerds/CoordinateSystems/CoordinateSystem/BaseCoordinateSystem.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[CoordinateSystemError](Coordinerds/CoordinateSystems/CoordinateSystem/CoordinateSystemError.md)   
</div>
   <div class="col" markdown="1">
[CompositeCoordinateSystem](Coordinerds/CoordinateSystems/CompositeCoordinateSystems/CompositeCoordinateSystem.md)   
</div>
   <div class="col" markdown="1">
[CompositeCoordinateSystemConverter](Coordinerds/CoordinateSystems/CompositeCoordinateSystems/CompositeCoordinateSystemConverter.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[CoordinateSet](Coordinerds/CoordinateSystems/CoordinateSet/CoordinateSet.md)   
</div>
   <div class="col" markdown="1">
[CoordinateTransform](Coordinerds/CoordinateTransformations/CoordinateTransform/CoordinateTransform.md)   
</div>
   <div class="col" markdown="1">
[TransformationFunction](Coordinerds/CoordinateTransformations/TransformationFunction/TransformationFunction.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[AffineTransform](Coordinerds/CoordinateTransformations/AffineTransform/AffineTransform.md)   
</div>
   <div class="col" markdown="1">
[TranslationTransform](Coordinerds/CoordinateTransformations/TranslationTransform/TranslationTransform.md)   
</div>
   <div class="col" markdown="1">
[RotationTransform](Coordinerds/CoordinateTransformations/RotationTransform/RotationTransform.md)   
</div>
</div>
  <div class="row">
   <div class="col" markdown="1">
[cartesian_to_zmatrix](Coordinerds/Conveniences/cartesian_to_zmatrix.md)   
</div>
   <div class="col" markdown="1">
[zmatrix_to_cartesian](Coordinerds/Conveniences/zmatrix_to_cartesian.md)   
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
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/examples/McUtils/Coordinerds.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/examples/McUtils/Coordinerds.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/ci/docs/McUtils/Coordinerds.md)/[New](https://github.com/McCoyGroup/McUtils/new/master/?filename=ci/docs/templates/McUtils/Coordinerds.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Coordinerds/__init__.py#L1?message=Update%20Docs)   
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