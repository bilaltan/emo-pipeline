from .phase0_ingestion import run_phase0
from .phase1_community import run_phase1, print_phase1_stats
from .phase2_subgraph import run_phase2
from .phase25_graph_store import run_phase25
from .phase26_training_blocks import run_phase26
from .phase27_block_audit import run_phase27
from .phase35_direct_block_training import run_phase35
from .phase36_sync_training import run_phase36
from .phase37_feature_propagation import run_phase37
from .phase38_edge_free_classifier import run_phase38
from .phase3_training import run_phase3
from .phase3b_caan import run_phase3b
from .phase4_baselines import (
    run_phase4,
    run_phase4b,
    run_phase4c,
    run_phase4d,
    run_phase4e,
    run_phase4f,
    run_phase4g,
    run_phase4h,
)
from .phase5_reporting import (
    print_accuracy_table,
    print_timing_table,
    save_plots_and_xlsx,
    print_summary,
)
