"""Pre-generate all query pools so parallel runs share identical data."""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqa.kgdata import load_graphs
from tqa.querygen import EPFO_TYPES
from tqa.train import get_train_pool, get_eval_pool

SEED = 42
TRAIN_PER_TYPE = 4000
EVAL_PER_TYPE = 500

if __name__ == "__main__":
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] pregen: loading graphs ...", flush=True)
    graphs = load_graphs()
    for n, g in graphs.items():
        print(f"[{time.strftime('%H:%M:%S')}]   {n}: |E|={g.num_ent} |R|={g.num_rel}", flush=True)
    for n, g in graphs.items():
        for qt in EPFO_TYPES:
            t1 = time.time()
            pool = get_train_pool(g, qt, TRAIN_PER_TYPE, SEED + 137 * (EPFO_TYPES.index(qt) + 1))
            print(f"[{time.strftime('%H:%M:%S')}] train pool {n}/{qt}: {len(pool)} "
                  f"({time.time()-t1:.0f}s)", flush=True)
    g = graphs["fb15k237"]
    for qt in EPFO_TYPES:
        t1 = time.time()
        pool = get_eval_pool(g, qt, EVAL_PER_TYPE, 777)
        print(f"[{time.strftime('%H:%M:%S')}] eval pool {qt}: {len(pool)} "
              f"({time.time()-t1:.0f}s)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] pregen DONE in {time.time()-t0:.0f}s", flush=True)
