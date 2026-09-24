"""run.py -- single entry point for module m5_progress.

Regenerates every table (output/tables/m5_progress_*), figure (output/figures/m5_progress_*) and processed file
(data/processed/m5_progress/*) of the module from raw data. Deterministic: all random draws use fixed seeds.
Uses at most 6 worker processes (see common.N_PROC). About 6,300 CPU-seconds: roughly 20 minutes of wall time on an idle
machine with 6 processes, 50+ minutes when the machine is heavily loaded.

    /Users/yigitokar/scaling-laws-pf/.venv/bin/python code/analysis/m5_progress/run.py [--only s2,s3,...] [--use-cache]

s1 (replication) always runs; --only restricts the later steps (s2 DMR incl. s2b multistart check, s3 attenuation, s4 allocative,
s5 CEG/Sahal, s6 Table 7); --use-cache reuses the cached Chinchilla/Farseer technology fits.
"""
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import N_PROC, PROC, log  # noqa: E402

CACHED = ('chinchilla_fits.json', 'farseer_chinchilla_fit.json', 'besiroglu_boot_theta.npy')


def main():
    only = None
    if '--only' in sys.argv:
        only = set(sys.argv[sys.argv.index('--only') + 1].split(','))
    t0 = time.time()
    if '--use-cache' not in sys.argv:   # regenerate intermediate fits from raw data (default)
        for f in CACHED:
            p = os.path.join(PROC, f)
            if os.path.exists(p):
                os.remove(p)
    ctx = mp.get_context('spawn')
    with ctx.Pool(N_PROC) as pool:
        import s1_replicate
        res1 = s1_replicate.run(pool)
        log(f"s1 done ({time.time() - t0:.0f}s)")
        if only is None or 's2' in only or 's2b' in only:
            import s2_dmr
            s2_dmr.multistart_check(pool, res1)     # is the published point the minimizer of Ho et al.'s objective?
        if only is None or 's2' in only:
            res2 = s2_dmr.run(pool, res1)
            log(f"s2 done ({time.time() - t0:.0f}s)")
        if only is None or 's3' in only:
            import s3_attenuation
            s3_attenuation.run(pool, res1)
            log(f"s3 done ({time.time() - t0:.0f}s)")
        if only is None or 's4' in only:
            import s4_allocative
            s4_allocative.run(pool, res1)
            log(f"s4 done ({time.time() - t0:.0f}s)")
        if only is None or 's5' in only:
            import s5_ceg_sahal
            s5_ceg_sahal.run(pool, res1)
            log(f"s5 done ({time.time() - t0:.0f}s)")
        if only is None or 's6' in only:
            import s6_table7
            s6_table7.run()
            log(f"s6 done ({time.time() - t0:.0f}s)")
    log(f"all done in {time.time() - t0:.0f}s")


if __name__ == '__main__':
    main()
