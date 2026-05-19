import json, random
from pathlib import Path

QRELS_JSONL = Path("data/qrels.jsonl")
OUT = Path("splits/split_seed42.json")

TRAIN_FRAC = 0.72
VAL_FRAC = 0.08
SEED = 42

def main():
    qids = set()
    with QRELS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            qids.add(o["qid"])
    qids = sorted(qids)

    random.Random(SEED).shuffle(qids)
    n = len(qids)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)

    split = {
        "seed": SEED,
        "train": qids[:n_train],
        "val": qids[n_train:n_train+n_val],
        "test": qids[n_train+n_val:]
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print("WROTE:", OUT.resolve(), "sizes:", len(split["train"]), len(split["val"]), len(split["test"]))

if __name__ == "__main__":
    main()
