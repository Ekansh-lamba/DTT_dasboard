"""famos -- a licence-free replacement for the imc FAMOS preprocessing chain.

Public surface:
    famos.ops       canonical pure operators (smo, filtlp, red)
    famos.chunked   streaming execution, bit-identical to unchunked
    famos.audit     per-run manifest for reproducibility
    famos.chain     YAML chain definition + runner
"""
from famos import ops                       # noqa: F401

__version__ = "0.1.0"
