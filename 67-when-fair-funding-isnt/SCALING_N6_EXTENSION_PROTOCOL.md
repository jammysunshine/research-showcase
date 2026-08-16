# Six-agent exact extension protocol

This is a post-hoc, exploratory extension of the preregistered three-agent
study. It freezes the same deterministic public-project model, values
`{0,1,2}`, normalized discrete critical payments, anonymity, DSIC, ex-post IR,
feasibility, no subsidy, and weak budget balance. The extension sets
`n_agents=6` and evaluates every cost `1..12`.

For each cost, the antichain enumerator exhausts all 128 anonymous monotone
allocation rules over the 28 sorted states. The primary verifier skips its
redundant permutation loop because allocation is keyed only by sorted states;
the serialized accepted rows are then replayed by the independent checker,
which explicitly checks every profile permutation. No rules are sampled or
tuned after inspecting outcomes.

Run with:

```bash
python3 scripts/run_n6_extension.py
```

The script records the complete accepted tables, a canonical SHA-256 digest,
independent failure count, elapsed wall time, and peak resident memory in
`artifacts/public_project_n6_extension.json`. The extension is evidence about
this finite six-agent lattice only; it is not an asymptotic characterization or
a result for randomized, subsidized, continuous, collusive, or unrestricted
payment mechanisms.
