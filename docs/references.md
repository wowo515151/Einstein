# Equation References and Conventions

This reference list is carried over from the inspected local `Lib4/References.txt` audit. The links provide traceability for the mathematical conventions; this project has not independently re-verified the linked pages during the current offline implementation.

| Topic | Reference | Convention used here |
| --- | --- | --- |
| Minkowski metric, index operations, tensor transformations, and contractions | [MIT OpenCourseWare 8.033, full lecture notes](https://ocw.mit.edu/courses/8-033-introduction-to-relativity-and-spacetime-physics-fall-2024/mit8_033_f24_lec_full.pdf) | Reorder components as needed to the time-first `(ct, x, y, z)` convention and signature `(-,+,+,+)`. |
| Four-velocity and four-momentum | [MIT OpenCourseWare 8.033, Lecture 7](https://ocw.mit.edu/courses/8-033-introduction-to-relativity-and-spacetime-physics-fall-2024/resources/mit8_033_f24_lec07_pdf/) | For massive particles, `u·u=-c²`, `p=(E/c,p)`, and `p·p=-m²c²`. |
| Lorentz boosts and invariant interval | [OpenStax, University Physics Volume 3, §5.5](https://openstax.org/books/university-physics-volume-3/pages/5-5-the-lorentz-transformation) | Translate equivalent coordinate conventions to the documented passive column-vector boost. |
| Minkowski inner product and Lorentz group | [University at Buffalo, Special Relativity Boot Camp](https://www.acsu.buffalo.edu/~whkinney/SpecialRelativityBootCamp/) | The library uses column vectors and checks `ΛᵀηΛ=η`. |
| Observer simultaneity slices and light-cone geometry | [MIT OpenCourseWare 8.033, Lecture 4](https://ocw.mit.edu/courses/8-033-introduction-to-relativity-and-spacetime-physics-fall-2024/resources/mit8_033_f24_lec04_pdf/); [University at Buffalo, Special Relativity Boot Camp](https://www.acsu.buffalo.edu/~whkinney/SpecialRelativityBootCamp/) | In an observer rest frame, a light-cone slice is a Euclidean sphere centered at the apex projection, with radius set by the temporal separation. |
| Electromagnetic potential, field tensor, Lorentz force, and retarded potentials | [Joachim Brod, University of Cincinnati, Electricity and Magnetism](https://homepages.uc.edu/~brodjm/teaching/lecture_notes/electrodynamics.pdf) | Field-tensor signs depend on conventions. This library follows the explicit Lib4 matrix with `F⁰ⁱ=Eᵢ/c`, `Fⁱʲ=εⁱʲᵏBₖ`, and `f^μ=qF^{μν}u_ν`. |
| Covariant Maxwell equations and electromagnetic stress-energy | [David Tong, General Relativity, §3](https://www.damtp.cam.ac.uk/user/tong/gr/grhtml/S3.html); [Richard Feynman, Lectures on Physics, Vol. II, Ch. 31](https://www.feynmanlectures.caltech.edu/II_31.html) | `J^μ=(cρ,j)`; the discrete solver uses a static spatial current with charge updated by discrete continuity. |
| Liénard–Wiechert potentials and retardation | [University of Texas at Austin, Classical Electromagnetism, §12.20](https://farside.ph.utexas.edu/teaching/jk1/Electromagnetism/node155.html) | Solve the past-null relation `c(t-tᵣ)=\|x-xₛ(tᵣ)\|` using the source position at retarded time. |
| Levi-Civita connection and Christoffel symbols | [David Tong, General Relativity, §3](https://www.damtp.cam.ac.uk/user/tong/gr/grhtml/S3.html) | `Γ^a_bc=½g^ad(∂_b g_dc+∂_c g_db-∂_d g_bc)`. |
| Covariant derivative, curvature, and Ricci contraction | [Sean Carroll, Lecture Notes on General Relativity, §§3–4](https://ned.ipac.caltech.edu/level5/March01/Carroll3/Carroll3.html) | The library documents its Riemann sign convention and uses `R_bd=R^a_bad`; other sign conventions are common. |

## Convention cross-checks

- With signature `(-+++)` and `x⁰=ct`, `a·b=-a⁰b⁰+Σᵢaⁱbⁱ`.
- The passive x-boost uses `ct′=γ(ct-βx)` and `x′=γ(x-βct)`.
- Proper-time-normalized four-velocity applies to massive timelike trajectories; photons have no rest frame.
- Hodge orientation is `ε⁰¹²³=+1`; in Lorentzian `(-+++)`, `ε₀₁₂₃=-1`, and the dual of a two-form obeys `**F=-F`.
- With first coordinate `ct`, the scalar d'Alembertian is `-∂²_(ct)+∇²`, equivalently `-c⁻²∂²_t+∇²`.
- The point-charge retarded potential is singular on the source worldline; the root finder requires a declared source-speed bound below `c`.
