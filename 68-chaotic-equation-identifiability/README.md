# The Chaos Paradox: Why Messier Data Can Mean Cleaner Equations

A multi-method, multi-system empirical study testing whether chaotic trajectories improve
recovery of governing equations from data (SINDy, symbolic regression, Koopman/EDMD),
across six system families with matched chaotic/non-chaotic regime pairs.

**Headline finding:** no single universal "chaos aids identifiability" law. Chaos
specifically (not just broad coverage) aids recovery for SINDy and symbolic regression,
replicated against both degenerate and rich non-chaotic controls with large effect sizes.
The same excitation that helps sparse recovery reverses outright for Koopman/EDMD dense
linear surrogates, for a mechanistically identified reason — though that reversal does not
hold universally against every rich non-chaotic control tested.

- [`Chaos_Paradox_Lay_Summary.pdf`](Chaos_Paradox_Lay_Summary.pdf) — plain-language summary, no math background needed
- [`PAPER.md`](PAPER.md) — full paper (abstract, methods, results, limitations, references)
- [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md) — short summary
- [`RESULTS.md`](RESULTS.md) / [`LIMITATIONS.md`](LIMITATIONS.md) / [`COUNTEREXAMPLES.md`](COUNTEREXAMPLES.md)
- [`REPRODUCIBILITY_MANIFEST.md`](REPRODUCIBILITY_MANIFEST.md) / [`REPLICATION_GUIDE.md`](REPLICATION_GUIDE.md) — how to rerun every result
- `src/`, `scripts/`, `experiments/`, `tests/`, `data/` — full code, raw result JSON, and test suite
