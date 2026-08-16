# Counterexamples and falsification results

The efficient sum-threshold rule at cost `c=3` is DSIC, ex-post IR, feasible, and anonymous, but fails weak budget balance. The first machine-readable witnesses are in `artifacts/public_project_study.json`; for example, report `(0,2,2)` induces critical payments `(0,1,1)`, totaling `2<3`.

The held-out audit evaluates sum-threshold rules on all `4^3=64` profiles with values `{0,1,2,3}`. It records budget-coverage failures for every threshold from 1 through 6 (207 failures in total). This is not hidden or relabeled as success: the searched three-level normalization does not generalize its cost-coverage guarantee to the held-out magnitude range.

The original Experiment 67 binary allocation audit and its three-agent extension remain in the repository as negative/verification baselines. They are not used as evidence for the public-project claim.
