## <a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator">RBFDInterpolator</a> 

<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators.py#L882)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators.py#L882?message=Update%20Docs)]
</div>

Provides a flexible RBF interpolator that also allows
for matching function derivatives







<div class="collapsible-section">
 <div class="collapsible-section collapsible-section-header" markdown="1">
## <a class="collapse-link" data-toggle="collapse" href="#methods" markdown="1"> Methods and Properties</a> <a class="float-right" data-toggle="collapse" href="#methods"><i class="fa fa-chevron-down"></i></a>
 </div>
 <div class="collapsible-section collapsible-section-body collapse show" id="methods" markdown="1">
 ```python
wendland_coefficient_cache: dict
poly_origin: float
InterpolationData: InterpolationData
Interpolator: Interpolator
```
<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.__init__" class="docs-object-method">&nbsp;</a> 
```python
__init__(self, pts, values, *derivatives, kernel: Union[<built-in function callable>, dict] = 'thin_plate_spline', kernel_options=None, auxiliary_basis=None, auxiliary_basis_options=None, extra_degree=0, clustering_radius=None, monomial_basis=True, multicenter_monomials=True, neighborhood_size=15, neighborhood_merge_threshold=None, neighborhood_max_merge_size=100, neighborhood_clustering_radius=None, solve_method='svd', max_condition_number=inf, error_threshold=0.01, bad_interpolation_retries=3, coordinate_transform=None, logger=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L888)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L888?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.gaussian" class="docs-object-method">&nbsp;</a> 
```python
gaussian(r, e=1, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1007)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1007?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.gaussian_derivative" class="docs-object-method">&nbsp;</a> 
```python
gaussian_derivative(n: int, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1012)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1012?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.gaussian_singularity_handler" class="docs-object-method">&nbsp;</a> 
```python
gaussian_singularity_handler(n: int, ndim: int, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1023)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1023?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.thin_plate_spline" class="docs-object-method">&nbsp;</a> 
```python
thin_plate_spline(r, o=3, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1050)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1050?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.thin_plate_spline_derivative" class="docs-object-method">&nbsp;</a> 
```python
thin_plate_spline_derivative(n: int, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1053)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1053?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.thin_plate_spline_singularity_handler" class="docs-object-method">&nbsp;</a> 
```python
thin_plate_spline_singularity_handler(n: int, ndim: int, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1068)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1068?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.wendland_coefficient" class="docs-object-method">&nbsp;</a> 
```python
wendland_coefficient(l, j, k): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1078)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1078?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.wendland_polynomial" class="docs-object-method">&nbsp;</a> 
```python
wendland_polynomial(r, d=None, k=3, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1100)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1100?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.wendland_polynomial_derivative" class="docs-object-method">&nbsp;</a> 
```python
wendland_polynomial_derivative(n: int, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1112)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1112?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.wendland_polynomial_singularity_handler" class="docs-object-method">&nbsp;</a> 
```python
wendland_polynomial_singularity_handler(n: int, ndim: int, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1125)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1125?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.zeros" class="docs-object-method">&nbsp;</a> 
```python
zeros(r, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1134)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1134?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.zeros_derivative" class="docs-object-method">&nbsp;</a> 
```python
zeros_derivative(n: int, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1137)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1137?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.zeros_singularity_handler" class="docs-object-method">&nbsp;</a> 
```python
zeros_singularity_handler(n: int, ndim: int, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1142)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1142?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.default_kernels" class="docs-object-method">&nbsp;</a> 
```python
@property
default_kernels(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1149)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1149?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.morse" class="docs-object-method">&nbsp;</a> 
```python
morse(r, a=1, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1177)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1177?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.morse_derivative" class="docs-object-method">&nbsp;</a> 
```python
morse_derivative(n: int, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1183)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1183?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.even_powers" class="docs-object-method">&nbsp;</a> 
```python
even_powers(r, o=1, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1194)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1194?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.even_powers_deriv" class="docs-object-method">&nbsp;</a> 
```python
even_powers_deriv(n: int, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1197)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1197?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.laguerre" class="docs-object-method">&nbsp;</a> 
```python
laguerre(r, k=3, shift=2.29428, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1207)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1207?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.laguerre_deriv" class="docs-object-method">&nbsp;</a> 
```python
laguerre_deriv(n: int, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1210)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1210?message=Update%20Docs)]
</div>
(-1)^n LaguerreL[k - n, n, x]


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.compact_laguerre" class="docs-object-method">&nbsp;</a> 
```python
compact_laguerre(r, e=1, k=3, shift=2.29428, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1218)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1218?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.compact_laguerre_deriv" class="docs-object-method">&nbsp;</a> 
```python
compact_laguerre_deriv(n: int, inds=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1221)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1221?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.default_auxiliary_bases" class="docs-object-method">&nbsp;</a> 
```python
@property
default_auxiliary_bases(self): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1240)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1240?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.evaluate_poly_matrix" class="docs-object-method">&nbsp;</a> 
```python
evaluate_poly_matrix(self, pts, degree, deriv_order=0, poly_origin=0.5, include_constant_term=True, monomials=True): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1285)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1285?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.evaluate_rbf_matrix" class="docs-object-method">&nbsp;</a> 
```python
evaluate_rbf_matrix(self, pts, centers, inds, deriv_order=0, zero_tol=1e-08): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1412)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1412?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.construct_matrix" class="docs-object-method">&nbsp;</a> 
```python
construct_matrix(self, pts, centers, inds, degree=0, deriv_order=0, zero_tol=1e-08, poly_origin=None, include_constant_term=True, force_square=False, monomials=True, multicentered_polys=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1479)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1479?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.svd_solve" class="docs-object-method">&nbsp;</a> 
```python
svd_solve(a, b, svd_cutoff=1e-12): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1514)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1514?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.solve_system" class="docs-object-method">&nbsp;</a> 
```python
solve_system(self, centers, vals, derivs: list, inds, solver=None, return_data=False, error_threshold=None): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1520)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1520?message=Update%20Docs)]
</div>


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.construct_evaluation_matrix" class="docs-object-method">&nbsp;</a> 
```python
construct_evaluation_matrix(self, pts, data, deriv_order=0): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1652)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1652?message=Update%20Docs)]
</div>

  - `pts`: `Any`
    > 
  - `data`: `Any`
    > 
  - `deriv_order`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.apply_interpolation" class="docs-object-method">&nbsp;</a> 
```python
apply_interpolation(self, pts, data, inds, reshape_derivatives=True, return_data=False, deriv_order=0): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1673)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1673?message=Update%20Docs)]
</div>

  - `pts`: `Any`
    > 
  - `data`: `Any`
    > 
  - `deriv_order`: `Any`
    > 
  - `:returns`: `_`
    >


<a id="McUtils.Zachary.NeighborBasedInterpolators.RBFDInterpolator.construct_interpolation" class="docs-object-method">&nbsp;</a> 
```python
construct_interpolation(self, inds, solver_data=False, return_error=False): 
```
<div class="docs-source-link" markdown="1">
[[source](https://github.com/McCoyGroup/McUtils/blob/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1719)/
[edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators/RBFDInterpolator.py#L1719?message=Update%20Docs)]
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
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/examples/McUtils/Zachary/NeighborBasedInterpolators/RBFDInterpolator.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/examples/McUtils/Zachary/NeighborBasedInterpolators/RBFDInterpolator.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/gh-pages/ci/docs/McUtils/Zachary/NeighborBasedInterpolators/RBFDInterpolator.md)/[New](https://github.com/McCoyGroup/McUtils/new/gh-pages/?filename=ci/docs/templates/McUtils/Zachary/NeighborBasedInterpolators/RBFDInterpolator.md)   
</div>
   <div class="col" markdown="1">
[Edit](https://github.com/McCoyGroup/McUtils/edit/master/Zachary/NeighborBasedInterpolators.py#L882?message=Update%20Docs)   
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