# Prior Art Map — Chaotic Equation Identifiability

Compiled 2026-08-16. Every entry below was retrieved via live WebSearch/WebFetch against the actual source; nothing is reconstructed from memory. Where only an abstract/secondary summary was accessible, that is noted.

---

## 1. The target paper: chaos and discoverability

**"When is a System Discoverable from Data? Discovery Requires Chaos"**
Zakhar Shumaylov, Peter Zaika, Philipp Scholl, Gitta Kutyniok, Lior Horesh, Carola-Bibiane Schönlieb. arXiv:2511.08860 (2025).
https://arxiv.org/abs/2511.08860

- **Definition of identifiability used:** a system is "discoverable" if its governing equations can be uniquely recovered from a single observed trajectory within a specified function space (continuous or analytic), i.e. no other admissible vector field in that class produces the same trajectory.
- **Central claim:** chaos — usually framed as an obstacle to prediction — is what makes a system's equations *uniquely* identifiable from a single trajectory. Systems that are chaotic everywhere on their domain are shown to be discoverable (in the continuous/analytic function-space sense) from one trajectory; the Lorenz system is used as the flagship example of analytic discoverability.
- **Negative counterpart:** systems with conserved quantities (first integrals) are *not* analytically identifiable from trajectory data alone — the conserved surface leaves a continuum of vector fields consistent with the same orbit. Non-chaotic (e.g., stable/periodic) systems generally need extra physical priors beyond the trajectory to pin down the equations.
- **Method:** theoretical/functional-analytic argument over spaces of continuous and analytic vector fields, rather than a new discovery algorithm; illustrated via canonical examples (Lorenz, systems with first integrals) rather than a large empirical benchmark.
- **Relevance to this project:** this is the specific claim PROMPT.md asks the lab to stress-test empirically (task 4: "systematically vary dynamical regime... test the hypothesis that richer/chaotic trajectories improve identifiability and identify exceptions"). It gives a theoretical anchor — discoverable-in-continuous/analytic-function-space is a much stronger and narrower notion than "SINDy recovers the right sparse coefficients from noisy finite data," so there is open room for a numerically-grounded companion or a counterexample at the finite-sample/model-class level. Note the paper's own scope limit: it argues within idealized (noise-free, single-trajectory, full function-space) settings — the project's mandate to test finite samples, noise, partial observation, and restricted libraries goes beyond what this paper establishes.

---

## 2. Sparse equation-discovery / SINDy literature

**"Discovering governing equations from data by sparse identification of nonlinear dynamical systems"**
Steven L. Brunton, Joshua L. Proctor, J. Nathan Kutz. *PNAS* 113(15), 3932–3937 (2016).
https://www.pnas.org/doi/10.1073/pnas.1517384113

- Introduces SINDy: builds a library of candidate nonlinear functions of the state, then uses sparse regression (sequential thresholded least squares) to select the few active terms, exploiting the empirical fact that most physical governing equations have only a handful of active terms in a suitable basis.
- Identifiability stance: recovery is *conditional on the true dynamics lying in the span of the candidate library* and on derivatives (or their surrogates) being estimable accurately from data — the paper does not claim identifiability guarantees outside that regime. This is the founding assumption this project must interrogate (library misspecification, near-duplicate terms, derivative-estimation leakage).

**"Sparse Identification of Nonlinear Dynamics with Control (SINDYc)"**
Steven L. Brunton, Joshua L. Proctor, J. Nathan Kutz. arXiv:1605.06682 (2016); related: E. Kaiser, J. N. Kutz, S. L. Brunton, "Sparse identification of nonlinear dynamics for model predictive control in the low-data limit," *Proc. R. Soc. A* 474(2219), 20180335 (2018).
https://arxiv.org/abs/1605.06682

- Extends SINDy to systems with exogenous control/forcing inputs, demonstrated on Lotka-Volterra and forced Lorenz systems; relevant to the project's "active experimental design restores identifiability" thread (task 6) since actuation is one lever for injecting persistent excitation.

**Weak-form SINDy (WSINDy)** — family of papers, e.g. D. A. Messenger, D. M. Bortz, "Weak SINDy for Partial Differential Equations," arXiv:2007.02848; extension "Weak-form modified sparse identification of nonlinear dynamics," arXiv:2410.17838 (2024).
https://arxiv.org/abs/2007.02848 , https://arxiv.org/abs/2410.17838

- Reformulates the SINDy regression in weak/integral form (test functions) instead of pointwise derivatives, giving orders-of-magnitude better robustness to measurement noise and avoiding explicit numerical differentiation — directly relevant to this project's requirement to separate "wrong equations" from "numerical differentiation leakage" (PROMPT.md task 9, independent critic mandate).

**"Ensemble-SINDy: Robust sparse model discovery in the low-data, high-noise limit, with active learning and control"**
U. Fasel, J. N. Kutz, B. W. Brunton, S. L. Brunton. arXiv:2111.10992; *Proc. R. Soc. A* 478(2260), 20210904 (2022).
https://arxiv.org/abs/2111.10992

- Applies bootstrap aggregation (bagging) over SINDy fits from data subsets to get inclusion probabilities per candidate term and uncertainty-quantified/probabilistic model discovery; used to identify models under noise levels roughly 2x worse than prior reports. Directly useful as a comparator baseline with built-in ambiguity/uncertainty output — relevant to quantifying "how ambiguous is recovery" rather than a binary pass/fail.

**"Benchmarking sparse system identification with low-dimensional chaos"**
A. A. Kaptanoglu, L. Zhang, Z. G. Nicolaou, U. Fasel, S. L. Brunton, et al. *Nonlinear Dynamics* 111(14), 13143–13164 (2023); arXiv:2302.10787.
https://arxiv.org/abs/2302.10787

- Large-scale benchmark of four SINDy-family optimizers across the Gilpin `dysts` standardized database of chaotic systems; finds the original sequential-thresholded-least-squares algorithm and a mixed-integer variant perform strongly, and weak-form SINDy improves results even on clean data. Directly relevant as (a) a prior large benchmark this project's 50+-regime benchmark should be positioned against/differentiated from, and (b) a ready source of matched chaotic system definitions (`dysts`) that could seed the system library.

---

## 3. Symbolic regression for dynamical systems

**"AI Feynman: a Physics-Inspired Method for Symbolic Regression"**
Silviu-Marian Udrescu, Max Tegmark. *Science Advances* 6(16), eaay2631 (2020); arXiv:1905.11481.
https://arxiv.org/abs/1905.11481

- Combines neural-network fitting with physics-motivated simplification heuristics (symmetry, separability, compositionality detection) to recursively simplify the symbolic-regression search; recovered all 100 Feynman-Lectures equations versus 71 for prior public tools, and improved success on a harder benchmark from 15% to 90%. Symbolic regression is noted as NP-hard in general — relevant to this project's "candidate function library / search coverage" completeness discussion (PROMPT.md's proof-obligations/search-coverage requirement for search-type projects).

**"Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl"**
Miles Cranmer et al. arXiv:2305.01582 (2023).
https://arxiv.org/abs/2305.01582

- Describes PySR (Python front-end) and its SymbolicRegression.jl backend: a multi-population evolutionary "evolve-simplify-optimize" search over symbolic expressions with distributed compute, positioned as a practical open-source alternative to gplearn-style genetic programming for scientific discovery. A natural third "family" alongside SINDy and Koopman/neural methods (PROMPT.md requirement 2: "at least three equation-discovery families").

**gplearn** (Trevor Stephens; scikit-learn-compatible genetic programming for symbolic regression).
https://github.com/trevorstephens/gplearn , https://gplearn.readthedocs.io/

- Established, lightweight open-source genetic-programming symbolic regressor (`SymbolicRegressor`) with scikit-learn API; commonly used as a simpler/faster baseline against PySR in the literature. Useful as the "cheap baseline" symbolic-regression comparator this project's resource ceiling likely favors.

---

## 4. Koopman operator / EDMD approaches

**"A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition"**
Matthew O. Williams, Ioannis G. Kevrekidis, Clarence W. Rowley. *J. Nonlinear Science* 25(6), 1307–1346 (2015).
https://link.springer.com/article/10.1007/s00332-015-9258-5

- Introduces EDMD: given snapshot pairs and a dictionary of scalar observables, approximates Koopman eigenvalues/eigenfunctions/modes via regression over the dictionary's span, without needing an explicit governing-equation model or access to the integrator. Serves as the project's natural "neural/Koopman" discovery family (requirement 2) and as a linear-in-observables alternative whose identifiability properties (dictionary completeness, spectral convergence) differ structurally from sparse-library SINDy — a candidate axis for showing method-dependent identifiability rather than a universal chaos-helps-identifiability law.

**"On Convergence of Extended Dynamic Mode Decomposition to the Koopman Operator"**
follow-up convergence analysis, arXiv:1703.04680; *J. Nonlinear Science* 28, 687–710 (2018).
https://arxiv.org/abs/1703.04680

- Establishes conditions (dictionary richness, sampling) under which EDMD's finite-dictionary approximation provably converges to the true (possibly infinite-dimensional) Koopman operator — directly analogous to the "sufficient conditions for identifiability in a restricted function class" theoretical contribution the project charter targets (requirement 7).

---

## 5. Theoretical identifiability / non-uniqueness results

**"Observability and Structural Identifiability of Nonlinear Biological Systems"**
Alejandro F. Villaverde. *Complexity* 2019, 8497093; arXiv:1812.04525.
https://arxiv.org/abs/1812.04525

- Tutorial/review using differential-geometry tools to jointly analyze *observability* (can internal state be inferred from outputs) and *structural identifiability* (can parameters be uniquely determined from noise-free, infinite-precision output data, given the model structure alone). Structural identifiability is explicitly a property of the model+observation map, independent of data quality — a formal vocabulary this project should adopt to distinguish "structurally non-identifiable" (no amount of clean data helps) from "practically non-identifiable" (finite/noisy data insufficient), which maps directly onto PROMPT.md's distinction between vector-field recovery and prediction accuracy.

**Persistent excitation and identifiability**
General system-identification literature (control-theory folklore, e.g. summarized via ScienceDirect topic overviews and Koopman-specific treatment: "Koopman Operators for Generalized Persistence of Excitation Conditions for Nonlinear Systems," arXiv:1906.10274).
https://arxiv.org/abs/1906.10274

- Persistent excitation (PE) is the standard condition guaranteeing that regressors span enough of state/input space over time for parameter estimates to converge uniquely; identifiability requires that distinct parameter sets cannot produce identical input-output behavior for *any* sufficiently rich signal, and PE operationalizes "rich enough." The Koopman-operator PE paper reformulates PE conditions (including for fixed vs. designed initial conditions) using Koopman operators — directly relevant as a candidate mechanism for *why* chaos might substitute for designed excitation: chaotic trajectories may satisfy PE-like richness conditions "for free" that periodic/quasiperiodic trajectories do not.

**Topological conjugacy and coordinate-transform ambiguity**
Standard dynamical-systems-theory concept (see e.g. "Mostly Conjugate: Relating Dynamical Systems — Beyond Homeomorphism," B. Webb et al.; "Let's Do the Time-Warp-Attend: Learning Topological Invariants of Dynamical Systems," arXiv:2312.09234).
https://arxiv.org/abs/2312.09234

- Two vector fields related by a smooth (or merely continuous) change of coordinates h, satisfying h∘f = g∘h, generate trajectories that are indistinguishable up to relabeling of state variables — a structural, exact (not approximate) source of non-identifiability whenever the observation operator does not pin down the coordinate system. This is a ready-made mechanism for the "adversarial indistinguishable alternatives... coordinate transforms" construction required in PROMPT.md task 5, and for building explicit counterexample families with a rigorous equivalence notion (conjugacy classes) rather than ad hoc curve-fitting coincidences.

- First integrals / conserved quantities as a non-identifiability mechanism are also treated directly by the target paper (arXiv:2511.08860, Section on first integrals) — noted here as the overlap point between areas 1 and 5: conserved quantities produce a continuum of trajectory-consistent vector fields, the same failure mode this theoretical literature frames via symmetries and lack of persistent excitation transverse to the conserved surface.

---

## Gaps and scope notes for the project

- No literature found that directly benchmarks the "chaos helps identifiability" claim empirically at finite sample size/noise against SINDy, symbolic regression, and Koopman side by side — this is open territory matching the project's stated smallest publishable unit.
- Structural identifiability theory (area 5) is mature for *parametric* ODE models with a fixed known structure (systems biology tradition); it is much less developed for the *nonparametric/library-search* setting SINDy and symbolic regression operate in. That gap is exactly where this project's contribution can land.
- `dysts` (Kaptanoglu et al. 2023 / Gilpin) is a ready, already-validated source of chaotic system definitions and could reduce the simulator-validation burden (PROMPT.md requirement 1) if license/scope permits reuse.
