# Data and artifact manifest

Do not commit private, restricted, or large raw datasets to Git.

For every external dataset or large artifact, record:

- canonical source and version/date;
- license and access conditions;
- exact acquisition or generation command;
- checksum and expected local path;
- preprocessing steps and split construction;
- known quality, privacy, leakage, and provenance limitations.

Small redistributable test fixtures may be committed when their source and license are documented. Prefer scripts that reconstruct derived data from an immutable source.
