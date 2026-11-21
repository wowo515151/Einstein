# Conventions and Units

Coordinates are `(ct, x, y, z)` with Minkowski signature `(-,+,+,+)`. A vector norm can be negative, zero, or positive. The default speed of light is exactly 299,792,458 m/s. SI units are used unless a context explicitly selects natural units with `c=1`. Four-velocity is normalized to `-c²`; four-momentum is `(E/c, px, py, pz)`.

Indices are explicitly marked `up` or `down` on `Tensor4`. Raising and lowering use the supplied metric. Electromagnetic tensor components use `F⁰ⁱ=Eᵢ/c` and the documented `(-,+,+,+)` convention. The differential geometry routines use finite differences; choose a step appropriate to the coordinate scale.
