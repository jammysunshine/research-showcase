# Reproducibility manifest

Environment: Python 3.14 (stdlib only), local CPU, no network/data download,
deterministic integer arithmetic. Repository commit and working-tree state are
recorded by Git.

Commands:

```bash
python3 -m unittest discover -s tests -v
python3 scripts/run_public_project_study.py
python3 scripts/verify_public_project_certificate.py
python3 scripts/run_value_extension.py
python3 scripts/run_n6_extension.py
python3 scripts/verify_scaling_theorem.py
python3 scripts/verify_value_lattice_theorem.py
python3 -m unittest tests.test_vcg_redistribution -v
python3 scripts/run_published_rule_audit.py
python3 scripts/run_guo_2019_baseline_audit.py
python3 scripts/run_guo_2016_baseline_audit.py
python3 scripts/run_guo_2019_three_agent_optimal_audit.py
python3 scripts/run_max_affine_certification.py
python3 scripts/verify_max_affine_certificate.py
python3 scripts/verify_source_network_certificates.py
python3 scripts/run_relu_benchmark.py
.venv/bin/python -m pip install -r requirements-verification.txt
.venv/bin/python scripts/verify_relu_benchmark_z3.py
python3 scripts/run_uniform_repair_study.py
.venv/bin/python scripts/verify_uniform_repair_z3.py
python3 scripts/run_repair_ir_tradeoff_study.py
.venv/bin/python scripts/verify_repair_ir_z3.py
python3 scripts/run_public_project_coalition_frontier.py
python3 scripts/verify_public_project_coalition_frontier.py
python3 scripts/run_public_project_coalition_scaling.py
python3 scripts/verify_public_project_coalition_scaling.py
python3 scripts/run_public_project_coalition_scaling_extended.py
python3 scripts/verify_public_project_coalition_scaling_extended.py
python3 scripts/run_public_project_coalition_value3_frontier.py
python3 scripts/verify_public_project_coalition_value3_frontier.py
python3 scripts/run_public_project_coalition_baseline_audit.py
python3 scripts/verify_public_project_coalition_baseline_audit.py
python3 scripts/run_public_project_false_name_audit.py
python3 scripts/verify_public_project_false_name_audit.py
python3 scripts/verify_public_project_coalition_lemma.py
python3 scripts/verify_public_project_coalition_characterization.py
python3 scripts/verify_public_project_coalition_characterization_extended.py
python3 -m unittest tests.test_public_project tests.test_model_and_verifier tests.test_public_project -v
```

The frozen Phase-IX scaling configuration is
`configs/phase_ix_relu_scaling.json` (SHA-256
`64f9c845ba60b5282d98aaab3b2f7bdc7eb01a05facb7bc7650c892c540aed60`). Run
`python3 scripts/run_relu_scaling_study.py` followed by
`python3 scripts/verify_relu_scaling_z3.py`. The source/compiler certificate
is `artifacts/phase_ix_relu_scaling_results.json` (SHA-256
`fd73f61e44dc510cf98bdaee436bf0a5f8a0dddeb7b50cecdb69c5e45ae49a14`), and
the exact-real challenge record is
`artifacts/phase_ix_relu_scaling_z3_certificate.json` (SHA-256
`6d0512dbf961c1a2cdf00d120fcbc49ca6793ed88a656077658a45adfb1a8197`). It
fixes five width-two sources across 3--7 agents and records 15 `unsat`
strict-improvement checks.

Expected summary: theorem construction checks for n=1..12 (806 constructed
mechanisms), with zero bounded replay failures; 16/32/64 candidates for `n=3,4,5`; accepted counts
`4,4,4,1,1,1`, `5,5,5,5,1,1,1,1`, and `6,6,6,6,6,1,1,1,1,1`; 74 cross-agent
rows; cross-agent independent failures 0; held-out failures 207.
The exploratory `n=3, max_value=3` extension has 66 candidates and accepted
counts `15,15,15,4,4,4,1,1,1`; all 60 serialized rows replay independently.
The exact six-agent extension has 128 candidates at each cost `1..12`, accepted
counts `7,7,7,7,7,7,1,1,1,1,1,1`, 48 serialized rows, and zero independent
replay failures. Its canonical digest is
`d13dfc940c38241ea21c4cc3f4abbbbda94fc012af643a1b9b4db0833963c5c7`.
The frozen Phase-VIII value-lattice confirmation uses
`configs/phase_viii_value_lattice_theorem.json` (SHA-256
`fa828ce7e3f4e5180b4a2751e9935615d1c47fd314fecb56bd3dfa7533b0868f`).
For `n=3`, `max_value=4`, and costs `1..12`, it requires exact predicted versus
exhaustive rule-set equality and independent replay of every accepted rule.
Counts are `65,65,65,15,15,15,4,4,4,1,1,1`: 255 rows total, zero independent
failures, digest
`977f91e2a4d34634f648a953d2450c12cac7ab636ea6bf97a76ae082d85979ec`.
The run took 24.00 seconds, used 45,842,432 bytes peak resident memory, and
emitted `artifacts/phase_viii_value_lattice_theorem.json` (1,248,674 bytes;
SHA-256 `c0c539a99d53d74fe34ae2ed1b8371d190686b58f53e04ac2403870aadd98a50`).

Artifacts are regenerated, not hand-edited:
`artifacts/public_project_study.json`, `artifacts/public_project_certificate.json`,
`artifacts/public_project_scaling.csv`, `artifacts/public_project_frontier.csv`,
`artifacts/public_project_value_extension.json`,
`artifacts/public_project_n6_extension.json`, and
`reports/public_project_frontier.svg`.
The Phase-VIII theorem confirmation is regenerated at
`artifacts/phase_viii_value_lattice_theorem.json`; its serialized rule tables
are intentionally retained so the independent replay target is inspectable.
The published-rule audit emits `artifacts/published_rule_audit.json`; the
frozen clean-run SHA-256 is
`4056655ed759dacaf561b36344b206745f98066e7d361bb78ed8a68bf50850df`.
The Phase III positive-control audit emits
`artifacts/guo_2019_three_agent_optimal_audit.json`; its SHA-256 is
`45df2f83d27eb70dd4874c7faf1ee3e764b8f4ab194c6df6269afce3a9b69a2b`.
The Phase IV generic shallow-max-affine certificate emits
`artifacts/max_affine_certification.json`; its SHA-256 is
`3bdb0c02b39107994f989894d1fcbe04157e0dce6814004e1cd11b217916f882`.
Its independent replay is `artifacts/max_affine_independent_certificate.json`;
its SHA-256 is `8b89a8ca8b4ed937f865be2b33c72f7d0d35518870c8bfbc3a9644bd13194f98`.
The direct source-network replay writes
`artifacts/max_affine_source_network_certificate.json`; it must match each
embedded public source network without reading a compiled expression.
Its frozen SHA-256 is
`1bcde48fc4f23e58a510027310fc723195a6553f23354472c904332111745d4f`.
The frozen Phase-V source/compiler crosscheck uses
`configs/relu_benchmark.json` (SHA-256
`64745342ba577353a2d4db81ebe9d67ac05a68b8a2d964b99cdeae69500d7b34`)
and emits `artifacts/relu_benchmark_results.json` (SHA-256
`8589d83fb5fcd4b3c9721ff4d3c2de2d7723c71a08fb5c46a518c88f96fd5cbd`).
It contains six deterministic rational fixtures over 3--5 agents; both exact
routes agree on every field after the retained zero-output-boundary bug fix.
The solver-backed strict-counterexample audit uses `z3-solver==5.0.0.0` from
`requirements-verification.txt` (SHA-256
`137fd7f42291439e332a9f3be1055e8d593de68f83048fee673164e0c4f0dc65`) and
writes `artifacts/relu_benchmark_z3_certificate.json` (SHA-256
`b9fb7e30792d9fe26eb7197b845a4c1ebd8a171c389713a260f574e5924f8545`). It
records 18 `unsat` strict-improvement queries plus direct rational witness
checks.
The Phase-VI uniform-repair study writes
`artifacts/uniform_repair_study.json` (SHA-256
`6978e7d21130f62136525abfc7e00a3f3df4af3442224263028b88f5ee71d3d2`);
the exact-real no-deficit replay writes
`artifacts/uniform_repair_z3_certificate.json` (SHA-256
`e074614506941fec5abd97360ce649ba1baf45d0e451ffa5f8992d703502d33d`).
It covers the disclosed four-agent decimal control plus all six frozen fixtures;
the scalar repair is `max(0,-s/n)` and each positive half-repair remains
deficit-producing at the original slack witness.
The largest frozen entry uses 22 arrangement planes, evaluates 7,315 exact
four-plane bases, and retains 116 feasible vertices.
The Phase-VII budget--IR study keeps the same corpus and writes
`artifacts/repair_ir_tradeoff_study.json` (SHA-256
`040e6a898ffe1d28c9053692280014c26ccbcc907f3c5e69d32023a93873d6a3`)
and `artifacts/repair_ir_tradeoff_z3_certificate.json` (SHA-256
`bd0adaa64c0d521c8f9a945e5687e26c16379ee622e0a0906d1e17ff960cd453`).
The direct source and compiler minima agree. Z3 records 28 `unsat`
strict-lower-bound queries and seven IR outcome queries; no repaired source is
ex-post IR in the declared efficient-Groves model.
The withdrawn Phase-VIII comparator-normalization audit is retained at
`artifacts/withdrawn_phase_viii_comparator_failure.json`, together with its
raw optimizer output and raw certificate. It is not regenerated or reported
as an experiment result: the candidate equals normalized pivotal VCG and the
record exists to prevent a repeat of the comparator error.

Hashes for the clean run:

```text
configs/public_project_study.json          58d3de51c22a1136a5af8445bd2999b0067b0f439624ffab689026caa3808e58
artifacts/public_project_study.json        069096fd620b7ec995e57619c1edee8024718a705cf4e8506846d8763672dbbb
artifacts/public_project_certificate.json  59d43612156d6800e2163c809015ac6bdc5bd726d58c3041d82c718f8adf6760
artifacts/public_project_scaling.csv       ed52a586503579a1c7c929c6faae5a557bfbd095c735649f838680dd034b6b4b
artifacts/public_project_frontier.csv      18bfe664fb845cc25bddb2f82368ed9864dc85a6bba2e1192cb9d0aa594bb3d5
reports/public_project_frontier.svg        fe2f6e1d346aee94b2470be42f0da5d2217f547058ce4c0f83547d7d7ba34b5e
artifacts/public_project_value_extension.json c2e4d513ba464dd2a5de02ed4e0eaea064287d963c40d6f12fe18cc3cbb779b2
artifacts/public_project_n6_extension.json  9ecd187b249b4bcefe5b25b0ea233339a101be5cd4d41f455364fde15f43c4d4
artifacts/public_project_scaling_theorem.json  97dad770e4ffa74532c5546a9f3c170b2e266f669c9ca68bc111781065e09c6d
artifacts/published_rule_audit.json       4056655ed759dacaf561b36344b206745f98066e7d361bb78ed8a68bf50850df
artifacts/guo_2019_grid_audit.json        61f4dc2efd184375e0ebd79a94dcecfe98a24005e0f78899087a56425ad48fb3
artifacts/guo_2016_grid_audit.json        92e3078d55320c4d6a9130cab16a07d0d264aef3db648244d61cb97adf7bdbf2
artifacts/guo_2019_three_agent_optimal_audit.json 45df2f83d27eb70dd4874c7faf1ee3e764b8f4ab194c6df6269afce3a9b69a2b
artifacts/max_affine_certification.json 3bdb0c02b39107994f989894d1fcbe04157e0dce6814004e1cd11b217916f882
artifacts/max_affine_independent_certificate.json 8b89a8ca8b4ed937f865be2b33c72f7d0d35518870c8bfbc3a9644bd13194f98
artifacts/max_affine_source_network_certificate.json 1bcde48fc4f23e58a510027310fc723195a6553f23354472c904332111745d4f
configs/relu_benchmark.json 64745342ba577353a2d4db81ebe9d67ac05a68b8a2d964b99cdeae69500d7b34
artifacts/relu_benchmark_results.json 8589d83fb5fcd4b3c9721ff4d3c2de2d7723c71a08fb5c46a518c88f96fd5cbd
requirements-verification.txt 137fd7f42291439e332a9f3be1055e8d593de68f83048fee673164e0c4f0dc65
artifacts/relu_benchmark_z3_certificate.json b9fb7e30792d9fe26eb7197b845a4c1ebd8a171c389713a260f574e5924f8545
artifacts/uniform_repair_study.json 6978e7d21130f62136525abfc7e00a3f3df4af3442224263028b88f5ee71d3d2
artifacts/uniform_repair_z3_certificate.json e074614506941fec5abd97360ce649ba1baf45d0e451ffa5f8992d703502d33d
artifacts/repair_ir_tradeoff_study.json 040e6a898ffe1d28c9053692280014c26ccbcc907f3c5e69d32023a93873d6a3
artifacts/repair_ir_tradeoff_z3_certificate.json bd0adaa64c0d521c8f9a945e5687e26c16379ee622e0a0906d1e17ff960cd453
configs/public_project_coalition_frontier.json 2aba936314ffa0a4d191fb12f1b77425498f811d5907b6f443c7c6330409ba4b
artifacts/public_project_coalition_frontier.json dda93a913c824f24004d2f538d328fbd6eb9799364dd710339fbba24c5769f3c
artifacts/public_project_coalition_frontier_certificate.json 187d5e97bb425fead0adb26626a141fca1704b7f64ea2d695b0ddd0aba2f991a
configs/public_project_coalition_scaling.json 83222573f7dbcabe83950ba238098abc012e98f478bd60f6cae8a57ae9fa286e
artifacts/public_project_coalition_scaling.json fd37a09b230d83047680a4ae3ec1433c8bd2d58afafcacda7fb7f7ad6d3065ce
artifacts/public_project_coalition_scaling_certificate.json 123609332c323d0b4dd3a7fd1092dd4079bbe284df99b709ea164c5c116180e7
configs/public_project_coalition_scaling_extended.json fdd4eb4ed41f8a4143f450e03261424ffd504ed6a0a7f4c8b728f0c6f786bc57
artifacts/public_project_coalition_scaling_extended.json 56258c86b27cd3e5b8fb184c6d0ee0e9d1de6a0e6645f61576533affc000a664
artifacts/public_project_coalition_scaling_extended_certificate.json 635867aa922d8aad3bfacd20c1864c12a9d91d8c9e027e096c295ea518d4672d
configs/public_project_coalition_value3_frontier.json 6403e640108a6ff24a5bbd5f6345c67a798a8e22243c5785779ef058ba5ed008
artifacts/public_project_coalition_value3_frontier.json d25cdbb9d3d403a4264152645286e49b2ecc5fef1d2b5303d16ca4aeef3de0a1
artifacts/public_project_coalition_value3_frontier_certificate.json 7fbabc205c77dab91f9efa85a241ead40ff61f54b8231dc12ba01697c0679304
configs/public_project_coalition_baseline_audit.json 27665716f17f80e7b7cbba8f124af427eee8991772f35e6482f992f66cde8417
artifacts/public_project_coalition_baseline_audit.json d18df1c01f30ff3037feb4c6dba04f3973ebc725f6c551b12a5aa6bf91ce4e3d
artifacts/public_project_coalition_baseline_audit_certificate.json 38644eabc5df0058ced4374e23209363bfcc2b5eaf26b4472ac1d44d2ee2dfeb
configs/public_project_false_name_audit.json 30030783a87479f8d877775058bf11d4645a0731f29f212bbc006f61dc2fcf3a
artifacts/public_project_false_name_audit.json 8b7626ed5cc1f25059cac382bf753e0bf7de513bce1f044c6132f1c7484875d3
artifacts/public_project_false_name_audit_certificate.json db4e2474464d0a38e0ea167354510d347514a5b8cfaddd429cbb9b7d59753332
artifacts/public_project_coalition_lemma_certificate.json fc030d4f60bc63f815224c070a4626b579f6c0a6efe8c91e079739d27a760db5
artifacts/public_project_coalition_characterization_certificate.json 9dd70ad48733e6be95cd8ff4b0f37e5638e7347ccb7d044ab0a2adf80ebe7be0
artifacts/public_project_coalition_characterization_extended_certificate.json c10e6217b03415820f95c65fbfbab7dc159796c8099441935961ec5e796e5f48
```
