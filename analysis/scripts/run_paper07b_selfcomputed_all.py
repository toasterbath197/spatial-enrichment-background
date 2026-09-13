"""
Convenience wrapper around run_paper07b_selfcomputed.py: runs all 8 sample
extractions followed by finalize, in one process. Only reasonable on a
machine with a few GB of free RAM and no per-command time cap - in this
project's own analysis sandbox, each step was run as its own separate command
instead (see run_paper07b_selfcomputed.py's docstring).
"""
import run_paper07b_selfcomputed as m

if __name__ == "__main__":
    for prefix in m.SAMPLES:
        m.extract(prefix)
    m.finalize()
