#!/usr/bin/env python3
"""
CLI script to run ablation experiments on Malicious-HDG using the cached base graph.
Ablations evaluated:
1. Full Graph (SAGE, transductive degree, CN cert key)
2. Without Certificate nodes & edges (no_certificates)
3. Certificate Co-issuance Key (coissue mode without common_name)
4. Without Nameserver edges (no_nameservers)
5. Without Registrar edges (no_registrars)
6. Without ASN edges (no_asn)
7. Graph-only vs Graph + Lexical features (graph_only_no_lexical)
8. Train-visible degree mode vs Transductive (degree_mode_train_visible)
9. Isolated Domain MLP (mlp_no_edges, zero message passing)
10. Attn Variant (HGTConv attention)
11. SAGE_Guard Clean (defense cost on unattacked graph)
Supports --smoke, --max-domains N, and --force for resumability.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.config import (
    load_config,
    get_resolved_paths,
    init_thread_pool,
    should_skip_run,
    save_run_result
)
from src.hdg.graph import build_base_graph, apply_split_scaling
from src.hdg.splits import random_stratified_split
from src.hdg.train import train_eval_gnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ablation Experiments on Cached Base Graph.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--single-seed", type=int, default=None, help="Run single seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--max-domains", type=int, default=None, help="Cap total domains")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cached runs")
    args = parser.parse_args()

    init_thread_pool()
    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, is_smoke=args.smoke, root_dir=REPO_ROOT)

    out_dir = paths.results_dir / "tables"
    paths.check_write_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = paths.processed_dir / "domains.parquet"
    graph_file = paths.processed_dir / "heterodata.pt"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Domains file not found: {parquet_file}")

    pfx = "[SMOKE] " if args.smoke else ("[FIXTURE] " if args.fixture else "[REAL] ")
    print(f"{pfx}Loading domain data and base graph...")
    df = pd.read_parquet(parquet_file)
    max_d = args.max_domains or (5000 if args.smoke else None)
    if max_d is not None and len(df) > max_d:
        df = df.sample(n=max_d, random_state=42).reset_index(drop=True)

    # Load cached base graph or build if missing
    if graph_file.exists() and not args.max_domains and not args.smoke:
        base_graph_cn = torch.load(graph_file, weights_only=False)
    else:
        cfg_cn = {**cfg, "graph": {**cfg.get("graph", {}), "cert_key_mode": "cn"}}
        base_graph_cn = build_base_graph(df, config=cfg_cn)

    # Base graph with co-issuance cert key
    cfg_co = {**cfg, "graph": {**cfg.get("graph", {}), "cert_key_mode": "coissue"}}
    base_graph_coissue = build_base_graph(df, config=cfg_co)

    if args.single_seed is not None:
        seeds = [args.single_seed]
    elif args.smoke:
        seeds = [42]
    else:
        seeds = cfg.get("randomness", {}).get("evaluation_seeds", [42, 1337, 2024, 777, 999])

    if args.epochs is not None:
        train_epochs = args.epochs
    elif args.smoke or args.fixture:
        train_epochs = 3
    else:
        train_epochs = cfg.get("model", {}).get("training", {}).get("epochs", 30)

    patience = 2 if (args.smoke or args.fixture) else cfg.get("model", {}).get("training", {}).get("patience", 10)

    # Ablation matrix definitions
    ablation_specs = {
        "full_sage": {
            "base": base_graph_cn, "include_certs": True, "drop_edge": None, "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage"
        },
        "no_certificates": {
            "base": base_graph_cn, "include_certs": False, "drop_edge": None, "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage"
        },
        "cert_coissue_key": {
            "base": base_graph_coissue, "include_certs": True, "drop_edge": None, "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage"
        },
        "no_nameservers": {
            "base": base_graph_cn, "include_certs": True, "drop_edge": "uses_ns", "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage"
        },
        "no_registrars": {
            "base": base_graph_cn, "include_certs": True, "drop_edge": "registered_by", "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage"
        },
        "no_asn": {
            "base": base_graph_cn, "include_certs": True, "drop_edge": "belongs_to_asn", "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage"
        },
        "graph_only_no_lexical": {
            "base": base_graph_cn, "include_certs": True, "drop_edge": None, "use_lex": False,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage"
        },
        "degree_mode_train_visible": {
            "base": base_graph_cn, "include_certs": True, "drop_edge": None, "use_lex": True,
            "degree_mode": "train_visible", "use_edges": True, "variant": "sage"
        },
        "mlp_no_edges": {
            "base": base_graph_cn, "include_certs": True, "drop_edge": None, "use_lex": True,
            "degree_mode": "transductive", "use_edges": False, "variant": "sage"
        },
        "attn_variant": {
            "base": base_graph_cn, "include_certs": True, "drop_edge": None, "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "attn"
        },
        "sage_guard_clean": {
            "base": base_graph_cn, "include_certs": True, "drop_edge": None, "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage_guard"
        }
    }

    results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ablation_specs.keys()}

    for name, a_spec in ablation_specs.items():
        print(f"\n=== Ablation: {name} ===")
        for s in seeds:
            run_name = f"ablation_{name}_seed_{s}"
            if should_skip_run(paths.results_dir, run_name, force=args.force):
                print(f"  [RESUME] Skipping run: {run_name}")
                with open(paths.results_dir / "runs" / f"{run_name}.json", "r", encoding="utf-8") as f:
                    r_cached = json.load(f)
                results[name].append(r_cached["metrics"])
                continue

            splits = random_stratified_split(df["label"].to_numpy(), seed=s)

            # Fast per-split scaling on cached base graph
            scaled_data, _ = apply_split_scaling(
                base=a_spec["base"],
                train_indices=splits.train_indices,
                val_indices=splits.val_indices,
                degree_mode=a_spec["degree_mode"],
                include_certificates=a_spec["include_certs"],
                drop_edge_type=a_spec["drop_edge"],
                use_lexical=a_spec["use_lex"]
            )

            in_channels = {nt: scaled_data[nt].x.shape[1] for nt in scaled_data.node_types}
            model = HeteroGNN(
                metadata=scaled_data.metadata(),
                in_channels_dict=in_channels,
                hidden_dim=64,
                out_dim=32,
                num_layers=2,
                variant=a_spec["variant"],
                use_edges=a_spec["use_edges"]
            )

            test_eval, _, _, _ = train_eval_gnn(
                model=model,
                data=scaled_data,
                splits=splits,
                df=df,
                epochs=train_epochs,
                patience=patience,
                seed=s
            )

            save_run_result(paths.results_dir, run_name, {"experiment": "ablation", "name": name, "seed": s, "metrics": test_eval})
            results[name].append(test_eval)
            print(f"  Seed {s} -> ROC-AUC: {test_eval['roc_auc']:.4f}, F1: {test_eval['f1']:.4f}")

    # Aggregate statistics
    aggregated_ablation: Dict[str, Any] = {}
    for name, runs in results.items():
        if not runs:
            continue
        aucs = [r["roc_auc"] for r in runs]
        f1s = [r["f1"] for r in runs]
        prs = [r["pr_auc"] for r in runs]
        aggregated_ablation[name] = {
            "num_seeds": len(runs),
            "roc_auc_mean": round(float(np.mean(aucs)), 4),
            "roc_auc_std": round(float(np.std(aucs)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "pr_auc_mean": round(float(np.mean(prs)), 4),
            "pr_auc_std": round(float(np.std(prs)), 4),
            "runs": runs
        }

    # Save JSON
    json_path = out_dir / "ablation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated_ablation, f, indent=2)

    # Markdown Summary
    pfx_t = "[FIXTURE - meaningless] " if args.fixture else ""
    md_lines = [
        f"# {pfx_t}Ablation Study Results",
        f"*Evaluated across {len(seeds)} seeded runs (mean ± std)*",
        "",
        "| Ablation Configuration | ROC-AUC | F1 Score | PR-AUC | Delta vs Full SAGE |",
        "|---|---|---|---|---|"
    ]
    base_auc = aggregated_ablation.get("full_sage", {}).get("roc_auc_mean", 0.0)
    for name, stat in aggregated_ablation.items():
        diff = stat["roc_auc_mean"] - base_auc
        md_lines.append(
            f"| `{name}` | {stat['roc_auc_mean']:.4f} ± {stat['roc_auc_std']:.4f} | {stat['f1_mean']:.4f} ± {stat['f1_std']:.4f} | {stat['pr_auc_mean']:.4f} ± {stat['pr_auc_std']:.4f} | {diff:+.4f} |"
        )
    md_lines.append("")

    md_path = out_dir / "ablation_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\n{pfx}=== ABLATION COMPLETE ===")
    print(f"Results JSON: {json_path}")
    print(f"Results Markdown: {md_path}")


if __name__ == "__main__":
    main()
