## data mixing w/ model swarms

use dirchlet sampler to train 50M @ 300M tok param GPTs on `text_mixing` sources

```bash
V=../.venv/bin/python
# full pipeline, backgrounded:
nohup bash -c '
  V=../.venv/bin/python
  $V swarm.py config  --exp topvig --pools cc_top_vig --n 1000   # 98-pool, 1000-model grid
  $V swarm.py launch  --exp topvig --wait                        # training array (idempotent)
  $V swarm.py reeval  --exp topvig --wait                        # score every ckpt on DEFAULT_SETS
  $V swarm.py collect --exp topvig                               # -> ../results/text_mixing/topvig/results.csv
  ../results/text_mixing/topvig/viz.py --exp topvig --all --spec --enrich  # figures (once viz.py is placed there)
' > run_topvig.log 2>&1 &

$V swarm.py status --exp topvig     # progress any time
```
