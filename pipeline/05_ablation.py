#!/usr/bin/env python3
"""
Malicious-HDG: Topological & Feature Ablation Study Orchestrator.

Scientific Purpose & Workflow:
------------------------------
Isolates the individual marginal contributions of heterogeneous graph topological edges,
domain lexical features, degree representation modes, and architectural components.
Evaluates 11 pre-committed ablation configurations across 5 evaluation seeds:

1. `full_sage`:
   - Complete HeteroData graph (GraphSAGE, transductive degree mode, CN certificate key, lexical features).
   - Serves as the primary reference baseline for all relative Delta evaluations.
2. `no_certificates`:
   - Drops all certificate nodes and (`domain`, `uses_cert`, `certificate`) edges.
   - Tests hypothesis: TLS certificates provide high-precision malicious clustering.
3. `cert_coissue_key`:
   - Switches certificate identity key from Common Name (CN) to Co-issuance key (hash of org, country,
     validity start/end, and valid length, omitting CN).
   - Tests detection of certificate syndication and bulk certificate issuance without shared names.
4. `no_nameservers`:
   - Drops all (`domain`, `uses_ns`, `nameserver`) edges.
   - Tests dependence on shared nameserver infrastructure (e.g. bulletproof DNS hosts).
5. `no_registrars`:
   - Drops all (`domain`, `registered_by`, `registrar`) edges.
   - Tests whether malicious campaigns concentrate in specific shady registrars.
6. `no_asn`:
   - Drops all (`domain`, `belongs_to_asn`, `asn`) edges.
   - Tests network-level autonomous system clustering and IP-to-ASN routing features.
7. `graph_only_no_lexical`:
   - Zeros out all lexical features (Shannon entropy, length, consonant ratio, digit ratio),
     retaining purely topological infrastructure relations and DNS flags.
   - Tests whether GNN performance relies on lexical artifacts vs relational cyber graph topology.
8. `degree_mode_train_visible`:
   - Scales domain node degree using train-visible degrees only, zeroing out degrees of unobserved nodes.
   - Eliminates transductive degree leakage for inductive deployment testing.
9. `mlp_no_edges`:
   - Drops all edges (zero message passing) while retaining tabular and lexical node features.
   - Quantifies the pure graph marginal value over isolated node-level deep learning.
10. `attn_variant`:
    - Evaluates Heterogeneous Attention (PyG HGTConv) with relational query-key-value projections.
11. `sage_guard_clean`:
    - Evaluates GNNGuard cosine edge pruning on unattacked clean data to measure benign defense overhead.

Memory & Resumption Safeguards:
-------------------------------
- Co-issuance base graph (`base_graph_coissue`) is built lazily on-demand and deallocated immediately.
- Explicit deallocation of PyTorch models, scaled graphs, and CUDA cache clearing after every run.
- Full idempotent resumption: completed runs in `results/real_gpu/runs/` are skipped automatically.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from src.hdg.models.hetero_gnn import HeteroGNN
from src.hdg.device import (
    resolve_device,
    get_execution_provenance,
    setup_shared_gpu_environment,
)
from src.hdg.config import (
    load_config,
    get_resolved_paths,
    init_thread_pool,
    should_skip_run,
    save_run_result
)
from src.hdg.data.graph import build_base_graph, apply_split_scaling
from src.hdg.data.splits import random_stratified_split
from src.hdg.training.train import train_eval_gnn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ablation Experiments on Cached Base Graph.")
    parser.add_argument("--fixture", action="store_true", help="Run against data_fixture/")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu', 'cuda', 'cuda:N')")
    parser.add_argument("--out-dir", type=str, default=None, help="Override results directory")
    parser.add_argument("--single-seed", type=int, default=None, help="Run single seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--max-domains", type=int, default=None, help="Cap total domains")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cached runs")
    args = parser.parse_args()

    init_thread_pool()
    cfg = load_config(args.config)
    paths = get_resolved_paths(cfg, is_fixture=args.fixture, is_smoke=args.smoke, root_dir=REPO_ROOT)

    device = resolve_device(args.device)
    is_cuda = (device.type == "cuda")
    if is_cuda:
        dev_idx = device.index if device.index is not None else 0
        mem_frac = float(os.environ.get("HDG_GPU_MEMORY_FRACTION", "0.95"))
        min_vram = float(os.environ.get("MIN_FREE_VRAM", "10.0"))
        setup_shared_gpu_environment(dev_idx=dev_idx, memory_fraction=mem_frac, min_free_vram_gb=min_vram)

    if args.out_dir is not None:
        custom_out = Path(args.out_dir)
        paths.results_dir = custom_out
    elif is_cuda and not (args.smoke or args.fixture):
        paths.results_dir = REPO_ROOT / "results" / "real_gpu"

    out_dir = paths.results_dir / "tables"
    paths.check_write_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = paths.processed_dir / "domains.parquet"
    graph_file = paths.processed_dir / "heterodata.pt"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Domains file not found: {parquet_file}")

    from src.hdg.data.graph import load_aligned_graph_cohort

    pfx = "[SMOKE] " if args.smoke else ("[FIXTURE] " if args.fixture else "[REAL] ")
    print(f"{pfx}Loading aligned domain data and base graph...")
    if graph_file.exists():
        base_graph_cn, df, manifest = load_aligned_graph_cohort(paths.processed_dir)
    else:
        df = pd.read_parquet(parquet_file)
        max_d = args.max_domains or (5000 if args.smoke else cfg.get("graph", {}).get("max_domains"))
        if max_d is not None and len(df) > max_d:
            df = df.sample(n=max_d, random_state=42).reset_index(drop=True)
        cfg_cn = {**cfg, "graph": {**cfg.get("graph", {}), "cert_key_mode": "cn"}}
        base_graph_cn = build_base_graph(df, config=cfg_cn)

    # Co-issuance base graph is allocated lazily only if cert_coissue_key runs
    base_graph_coissue = None

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
        train_epochs = cfg.get("model", {}).get("training", {}).get("epochs", 200)

    # 50-epoch floor and 60-epoch patience adhering to Phase 2b specifications
    if args.smoke or args.fixture:
        patience = 2
        min_epochs = 1
    else:
        patience = 60
        min_epochs = 50

    # Ablation matrix definitions with lazy base graph binding
    ablation_specs = {
        "full_sage": {
            "base_type": "cn", "include_certs": True, "drop_edge": None, "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage"
        },
        "no_certificates": {
            "base_type": "cn", "include_certs": False, "drop_edge": None, "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage"
        },
        "cert_coissue_key": {
            "base_type": "coissue", "include_certs": True, "drop_edge": None, "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage"
        },
        "no_nameservers": {
            "base_type": "cn", "include_certs": True, "drop_edge": "uses_ns", "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage"
        },
        "no_registrars": {
            "base_type": "cn", "include_certs": True, "drop_edge": "registered_by", "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage"
        },
        "no_asn": {
            "base_type": "cn", "include_certs": True, "drop_edge": "belongs_to_asn", "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage"
        },
        "graph_only_no_lexical": {
            "base_type": "cn", "include_certs": True, "drop_edge": None, "use_lex": False,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage"
        },
        "degree_mode_train_visible": {
            "base_type": "cn", "include_certs": True, "drop_edge": None, "use_lex": True,
            "degree_mode": "train_visible", "use_edges": True, "variant": "sage"
        },
        "mlp_no_edges": {
            "base_type": "cn", "include_certs": True, "drop_edge": None, "use_lex": True,
            "degree_mode": "transductive", "use_edges": False, "variant": "sage"
        },
        "attn_variant": {
            "base_type": "cn", "include_certs": True, "drop_edge": None, "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "attn"
        },
        "sage_guard_clean": {
            "base_type": "cn", "include_certs": True, "drop_edge": None, "use_lex": True,
            "degree_mode": "transductive", "use_edges": True, "variant": "sage_guard"
        }
    }

    results: Dict[str, List[Dict[str, Any]]] = {name: [] for name in ablation_specs.keys()}

    for name, a_spec in ablation_specs.items():
        print(f"\n=== Ablation: {name} ===")

        # Resolve base graph with lazy allocation for co-issuance mode
        if a_spec.get("base_type") == "coissue":
            needs_coissue_run = any(
                not should_skip_run(paths.results_dir, f"ablation_{name}_seed_{s}", force=args.force)
                for s in seeds
            )
            if needs_coissue_run and base_graph_coissue is None:
                print(f"  [MEMORY OPT] Lazily building co-issuance base graph on demand for '{name}'...")
                cfg_co = {**cfg, "graph": {**cfg.get("graph", {}), "cert_key_mode": "coissue"}}
                base_graph_coissue = build_base_graph(df, config=cfg_co)
            target_base = base_graph_coissue
        else:
            target_base = base_graph_cn

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
                base=target_base,
                train_indices=splits.train_indices,
                val_indices=splits.val_indices,
                degree_mode=a_spec["degree_mode"],
                include_certificates=a_spec["include_certs"],
                drop_edge_type=a_spec["drop_edge"],
                use_lexical=a_spec["use_lex"]
            )

            # Explicit seeding before instantiation for reproducible initialization
            torch.manual_seed(s)
            np.random.seed(s)
            random.seed(s)

            in_channels = {nt: scaled_data[nt].x.shape[1] for nt in scaled_data.node_types}
            m_cfg = cfg.get("model", {})
            tr_cfg = m_cfg.get("training", {})
            p_nt = "fqdn" if "fqdn" in scaled_data.node_types else "domain"
            has_cb = hasattr(scaled_data[p_nt], "char_bytes") and scaled_data[p_nt].char_bytes is not None
            model = HeteroGNN(
                metadata=scaled_data.metadata(),
                in_channels_dict=in_channels,
                hidden_dim=64,
                out_dim=32,
                num_layers=2,
                variant=a_spec["variant"],
                use_edges=a_spec["use_edges"],
                use_gated_agg=bool(m_cfg.get("sage", {}).get("use_gated_agg", True)),
                use_char_cnn=bool(m_cfg.get("use_char_cnn", False) and a_spec["use_lex"] and has_cb),
                char_cnn_dim=int(m_cfg.get("char_cnn_dim", 16)),
                use_completeness_gate=bool(m_cfg.get("use_completeness_gate", False)),
            )

            min_epochs = 1 if (args.smoke or args.fixture) else 50
            test_eval, _, _, _ = train_eval_gnn(
                model=model,
                data=scaled_data,
                splits=splits,
                df=df,
                epochs=train_epochs,
                patience=patience,
                min_epochs=min_epochs,
                device=device,
                seed=s,
                use_class_weights=bool(tr_cfg.get("use_class_weights", False)),
                loss_type=str(tr_cfg.get("loss_type", "ce")),
                focal_alpha=float(tr_cfg.get("focal_alpha", 0.75)),
                focal_gamma=float(tr_cfg.get("focal_gamma", 2.0)),
                effective_beta=float(tr_cfg.get("effective_beta", 0.9999)),
                threshold_criterion=str(tr_cfg.get("threshold_criterion", "f1")),
                max_validation_fpr=float(tr_cfg.get("max_validation_fpr", 0.015)),
                use_temperature_scaling=bool(tr_cfg.get("use_temperature_scaling", False)),
            )

            run_payload = {
                "experiment": "ablation",
                "name": name,
                "seed": s,
                "metrics": test_eval,
                "provenance": get_execution_provenance(device)
            }
            save_run_result(paths.results_dir, run_name, run_payload)
            results[name].append(test_eval)
            print(f"  Seed {s} -> ROC-AUC: {test_eval['roc_auc']:.4f}, F1: {test_eval['f1']:.4f}")

            del model, scaled_data
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Free lazily allocated co-issuance base graph to reclaim host RAM
        if a_spec.get("base_type") == "coissue" and base_graph_coissue is not None:
            print("  [MEMORY OPT] Deallocating co-issuance base graph to reclaim host RAM...")
            del base_graph_coissue
            base_graph_coissue = None
            import gc
            gc.collect()

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
