"""Download the Santa Fe Time Series Competition Dataset A (chaotic
far-infrared laser) from a Wayback Machine mirror of its original host
(Andreas Weigend's Stanford page, now offline). See NEXT_STEPS.md item #9,
DECISION_LOG.md "Real-world dataset for SINDy transfer check", SOURCES.json.

Two direct alternative sources were tried and rejected before this one:
  1. PhysioNet's "santa-fe" collection (https://physionet.org/content/santa-fe/)
     -- this is Santa Fe Dataset B (physiological ECG/respiration data), a
     DIFFERENT dataset from the same competition, not Dataset A.
  2. CRAN/GitHub `TSPred` R package's bundled `SantaFe.A` data -- genuinely
     has the data, but only via R, not a plain public URL; would require
     installing R purely to extract a 1000-line text file, more friction
     than the working Wayback Machine URL below for no provenance benefit.

Source actually used: web.archive.org's 2016-04-24 snapshot of
http://www-psych.stanford.edu/~andreas/Time-Series/SantaFe/{A.dat,A.cont}
(the original live host is offline; this is the earliest complete snapshot
found). A.dat = the original 1000-point competition training series;
A.cont = 9093 further points from the same laser run (the competition's
answer-key continuation was only the first 100 of these, but the full
continuation was also archived) -- used here as an independent-in-time
confirmation series, NOT an independent-initial-condition trajectory (see
DECISION_LOG.md for why this distinction matters for this project's claims).
"""
import hashlib
import os
import urllib.request

BASE = "https://web.archive.org/web/20160424015114id_/http://www-psych.stanford.edu/~andreas/Time-Series/SantaFe/"
FILES = ["A.dat", "A.cont"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "santafe_laser")

EXPECTED_SHA256 = {
    "A.dat": "14faec41629ead4f45d9c055a4494269dd9122dd559594502c0c522dc98a9eb2",
    "A.cont": "feebf2460cb8c945b87dde368cccde95a05ee06955827c4bec93f2043db6ec5e",
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for fname in FILES:
        url = BASE + fname
        dest = os.path.join(OUT_DIR, fname)
        print(f"Downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        with open(dest, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        expected = EXPECTED_SHA256[fname]
        status = "OK" if digest == expected else "MISMATCH"
        print(f"  sha256={digest} expected={expected} [{status}]")
        if status != "OK":
            print(f"  WARNING: checksum mismatch for {fname} -- source may have changed.")


if __name__ == "__main__":
    main()
