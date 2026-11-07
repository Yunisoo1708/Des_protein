"""Run RFD4's run_alphafold_batch.py on stock AF3: drop the container-only `of3_weights` kwarg (only allowed when False)."""
import functools, os, runpy, sys
from alphafold3.data import featurisation
_orig = featurisation.featurise_input
@functools.wraps(_orig)
def featurise_input(*a, **kw):
    if kw.pop("of3_weights", False):
        raise RuntimeError("OpenFold3 weights need the patched AF3 fork; stock AF3 cannot run them")
    return _orig(*a, **kw)
featurisation.featurise_input = featurise_input
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
