"""Fixed-cap synthetic GPU stage curves; --cpu-smoke checks local model wiring."""

import argparse
import json
from pathlib import Path

from looped_cdb.benchmarks.synthetic_stages import SyntheticStageConfig, cpu_stage_smoke, profile_stages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/feedback/stage_costs.json"))
    for name in (
        "resident_cap",
        "context_length",
        "hidden_size",
        "num_heads",
        "intermediate_size",
        "vocab_size",
        "prelude_layers",
        "core_layers",
        "coda_layers",
        "warmup",
        "repeats",
        "seed",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(SyntheticStageConfig(), name))
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--backend", choices=("flash_attention_3", "flash_attention_2"), default="flash_attention_3")
    parser.add_argument("--eager", action="store_true", help="disable decode graphs, keeping the same resident cap")
    args = vars(parser.parse_args())
    smoke, output, eager = args.pop("cpu_smoke"), args.pop("output"), args.pop("eager")
    if smoke:
        print(json.dumps(cpu_stage_smoke(), indent=2))
        return
    result = profile_stages(SyntheticStageConfig(**args, cuda_graph=not eager))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
