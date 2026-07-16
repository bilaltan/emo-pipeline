import os
import time
import pandas as pd

def print_accuracy_table(datasets, algorithms, phase3_results, phase4_results, phase4b_results=None,
                         phase4c_results=None, phase4d_results=None, phase4e_results=None,
                         phase4f_results=None, phase4g_results=None, phase4h_results=None, phase3b_results=None, gnn_models=None):
    """Per-algorithm performance comparison including node acc and link AUC."""
    print("\n" + "="*95)
    print("  PHASE 5A — PERFORMANCE COMPARISON (NODE ACCURACY & LINK AUC)")
    print("="*95)
    
    if gnn_models is None:
        gnn_models = ['sage', 'gat', 'gatv2', 'transformer', 'clusterscl']
    
    for dataset in datasets:
        print(f"\n  Dataset: {dataset}")
        hdr = (f"  {'_Algorithm':<18} {'Node Acc':>10} {'Comm Acc':>10} "
               f"{'Bnd Acc':>10} {'Int Acc':>10} {'Link AUC':>12}")
        print(hdr)
        print("  " + "-"*75)
        
        # Print partitioned results
        for alg in algorithms:
            for m_type in gnn_models:
                key = (dataset, alg, m_type)
                df = phase3_results.get(key)
                if df is None and m_type == 'sage':
                    df = phase3_results.get((dataset, alg))
                if df is not None:
                    g_acc = df.attrs.get('weighted_comm_acc', float('nan'))
                    c_acc = df['comm_test_acc'].mean()
                    b_acc = df[df['n_boundary'] > 0]['boundary_acc'].mean() if len(df[df['n_boundary'] > 0]) > 0 else 0.0
                    i_acc = df[df['n_internal'] > 0]['internal_acc'].mean() if len(df[df['n_internal'] > 0]) > 0 else 0.0
                    l_auc = df.attrs.get('weighted_comm_link_auc', df['comm_link_auc'].mean() if 'comm_link_auc' in df.columns else float('nan'))
                    label = f"{alg}-{m_type}"
                    print(f"  {label:<18} {g_acc:>10.4f} {c_acc:>10.4f} {b_acc:>10.4f} {i_acc:>10.4f} {l_auc:>12.4f}")
                
                # Phase 3b CaaN GRL
                if phase3b_results is not None:
                    dfb = phase3b_results.get(key)
                    if dfb is not None:
                        gb_acc = dfb.attrs.get('weighted_comm_acc', float('nan'))
                        cb_acc = dfb['comm_test_acc'].mean()
                        bb_acc = dfb[dfb['n_boundary'] > 0]['boundary_acc'].mean() if len(dfb[dfb['n_boundary'] > 0]) > 0 else 0.0
                        ib_acc = dfb[dfb['n_internal'] > 0]['internal_acc'].mean() if len(dfb[dfb['n_internal'] > 0]) > 0 else 0.0
                        lb_auc = dfb.attrs.get('weighted_comm_link_auc', dfb['comm_link_auc'].mean() if 'comm_link_auc' in dfb.columns else float('nan'))
                        labelb = f"{alg}-{m_type}-caan"
                        print(f"  {labelb:<18} {gb_acc:>10.4f} {cb_acc:>10.4f} {bb_acc:>10.4f} {ib_acc:>10.4f} {lb_auc:>12.4f}")
                
        # Print global baselines
        bl = phase4_results.get(dataset)
        if bl:
            bl_acc = bl.get('test_acc', float('nan'))
            bl_auc = bl.get('link_auc', float('nan'))
            print(f"  {'SAGE-BL':<18} {bl_acc:>10.4f} {'—':>10} {'—':>10} {'—':>10} {bl_auc:>12.4f}")
        if phase4b_results:
            b4b = phase4b_results.get(dataset)
            if b4b:
                print(f"  {'DISTDGL':<18} {b4b.get('test_acc', float('nan')):>10.4f} {'—':>10} {'—':>10} {'—':>10} {b4b.get('link_auc', float('nan')):>12.4f}")
        if phase4c_results:
            b4c = phase4c_results.get(dataset)
            if b4c:
                print(f"  {'ARMA-BL':<18} {b4c.get('test_acc', float('nan')):>10.4f} {'—':>10} {'—':>10} {'—':>10} {b4c.get('link_auc', float('nan')):>12.4f}")
        if phase4d_results:
            b4d = phase4d_results.get(dataset)
            if b4d:
                print(f"  {'ASAP-BL':<18} {b4d.get('test_acc', float('nan')):>10.4f} {'—':>10} {'—':>10} {'—':>10} {b4d.get('link_auc', float('nan')):>12.4f}")
        if phase4e_results:
            b4e = phase4e_results.get(dataset)
            if b4e:
                print(f"  {'GAT-BL':<18} {b4e.get('test_acc', float('nan')):>10.4f} {'—':>10} {'—':>10} {'—':>10} {b4e.get('link_auc', float('nan')):>12.4f}")
        if phase4f_results:
            b4f = phase4f_results.get(dataset)
            if b4f:
                print(f"  {'TRANS-BL':<18} {b4f.get('test_acc', float('nan')):>10.4f} {'—':>10} {'—':>10} {'—':>10} {b4f.get('link_auc', float('nan')):>12.4f}")
        if phase4g_results:
            b4g = phase4g_results.get(dataset)
            if b4g:
                print(f"  {'CL-SCL-BL':<18} {b4g.get('test_acc', float('nan')):>10.4f} {'—':>10} {'—':>10} {'—':>10} {b4g.get('link_auc', float('nan')):>12.4f}")
        if phase4h_results:
            b4h = phase4h_results.get(dataset)
            if b4h:
                print(f"  {'GATv2-BL':<18} {b4h.get('test_acc', float('nan')):>10.4f} {'—':>10} {'—':>10} {'—':>10} {b4h.get('link_auc', float('nan')):>12.4f}")

        # Size-bucket breakdown per algorithm-model
        print(f"\n  [{dataset}] Performance by community size bucket:")
        for alg in algorithms:
            for m_type in gnn_models:
                key = (dataset, alg, m_type)
                df = phase3_results.get(key)
                if df is None and m_type == 'sage':
                    df = phase3_results.get((dataset, alg))
                if df is not None:
                    print(f"    {alg}-{m_type}:")
                    for bucket in ['small', 'medium', 'large']:
                        sub = df[df['size_bucket'] == bucket]
                        if len(sub):
                            sub_acc = sub['comm_test_acc'].mean()
                            sub_auc = sub['comm_link_auc'].mean() if 'comm_link_auc' in sub.columns else 0.5
                            print(f"      {bucket:<8}: n={len(sub):,}  "
                                  f"acc={sub_acc:.4f}  auc={sub_auc:.4f}")
                if phase3b_results is not None:
                    dfb = phase3b_results.get(key)
                    if dfb is not None:
                        print(f"    {alg}-{m_type}-caan:")
                        for bucket in ['small', 'medium', 'large']:
                            sub = dfb[dfb['size_bucket'] == bucket]
                            if len(sub):
                                sub_acc = sub['comm_test_acc'].mean()
                                sub_auc = sub['comm_link_auc'].mean() if 'comm_link_auc' in sub.columns else 0.5
                                print(f"      {bucket:<8}: n={len(sub):,}  "
                                      f"acc={sub_acc:.4f}  auc={sub_auc:.4f}")


def print_timing_table(datasets, algorithms, timing, gnn_models=None):
    """Per-algorithm timing breakdown."""
    print("\n" + "="*80)
    print("  PHASE 5B — TIMING COMPARISON")
    print("="*80)
    if gnn_models is None:
        gnn_models = ['sage', 'gat', 'transformer', 'clusterscl']
    for dataset in datasets:
        print(f"\n  Dataset: {dataset}")
        hdr = (f"  {'Algorithm':<18} {'Phase1':>9} {'Phase2':>9} "
               f"{'Phase3':>9} {'Phase3b':>9} {'Total':>9}")
        print(hdr)
        print("  " + "-"*68)
        for alg in algorithms:
            for m_type in gnn_models:
                t1 = timing.get(('phase1', dataset, alg), 0.0)
                t2 = timing.get(('phase2', dataset, alg), 0.0)
                t3 = timing.get(('phase3', dataset, alg, m_type), 0.0)
                if t3 == 0.0 and m_type == 'sage':
                    t3 = timing.get(('phase3', dataset, alg), 0.0)
                t3b = timing.get(('phase3b', dataset, alg, m_type), 0.0)
                if t3 > 0.0 or t3b > 0.0 or (m_type == 'sage' and (t1 > 0.0 or t2 > 0.0)):
                    total = t1 + t2 + t3 + t3b
                    label = f"{alg}-{m_type}"
                    print(f"  {label:<18} {t1:>8.1f}s {t2:>8.1f}s {t3:>8.1f}s {t3b:>8.1f}s {total:>8.1f}s")
        t4 = timing.get(('phase4', dataset), 0)
        t4_node = timing.get(('phase4_node', dataset), 0)
        t4_link = timing.get(('phase4_link', dataset), 0)
        print(f"  {'BASELINE (SAGE)':<18} {'—':>9} {'—':>9} {'—':>9} {t4:>8.1f}s (Node: {t4_node:.1f}s, Link: {t4_link:.1f}s)")
        
        t4b = timing.get(('phase4b', dataset), None)
        if t4b is not None:
            t4b_node = timing.get(('phase4b_node', dataset), 0)
            t4b_link = timing.get(('phase4b_link', dataset), 0)
            print(f"  {'DISTDGL':<18} {'—':>9} {'—':>9} {'—':>9} {t4b:>8.1f}s (Node: {t4b_node:.1f}s, Link: {t4b_link:.1f}s)")
            
        t4c = timing.get(('phase4c', dataset), None)
        if t4c is not None:
            t4c_node = timing.get(('phase4c_node', dataset), 0)
            t4c_link = timing.get(('phase4c_link', dataset), 0)
            print(f"  {'ARMA':<18} {'—':>9} {'—':>9} {'—':>9} {t4c:>8.1f}s (Node: {t4c_node:.1f}s, Link: {t4c_link:.1f}s)")
            
        t4d = timing.get(('phase4d', dataset), None)
        if t4d is not None:
            t4d_node = timing.get(('phase4d_node', dataset), 0)
            t4d_link = timing.get(('phase4d_link', dataset), 0)
            print(f"  {'ASAP':<18} {'—':>9} {'—':>9} {'—':>9} {t4d:>8.1f}s (Node: {t4d_node:.1f}s, Link: {t4d_link:.1f}s)")

        t4e = timing.get(('phase4e', dataset), None)
        if t4e is not None:
            t4e_node = timing.get(('phase4e_node', dataset), 0)
            t4e_link = timing.get(('phase4e_link', dataset), 0)
            print(f"  {'GAT-BL':<18} {'—':>9} {'—':>9} {'—':>9} {t4e:>8.1f}s (Node: {t4e_node:.1f}s, Link: {t4e_link:.1f}s)")

        t4f = timing.get(('phase4f', dataset), None)
        if t4f is not None:
            t4f_node = timing.get(('phase4f_node', dataset), 0)
            t4f_link = timing.get(('phase4f_link', dataset), 0)
            print(f"  {'TRANS-BL':<18} {'—':>9} {'—':>9} {'—':>9} {t4f:>8.1f}s (Node: {t4f_node:.1f}s, Link: {t4f_link:.1f}s)")

        t4g = timing.get(('phase4g', dataset), None)
        if t4g is not None:
            t4g_node = timing.get(('phase4g_node', dataset), 0)
            t4g_link = timing.get(('phase4g_link', dataset), 0)
            print(f"  {'CL-SCL-BL':<18} {'—':>9} {'—':>9} {'—':>9} {t4g:>8.1f}s (Node: {t4g_node:.1f}s, Link: {t4g_link:.1f}s)")

        t4h = timing.get(('phase4h', dataset), None)
        if t4h is not None:
            t4h_node = timing.get(('phase4h_node', dataset), 0)
            t4h_link = timing.get(('phase4h_link', dataset), 0)
            print(f"  {'GATv2-BL':<18} {'—':>9} {'—':>9} {'—':>9} {t4h:>8.1f}s (Node: {t4h_node:.1f}s, Link: {t4h_link:.1f}s)")


def save_plots_and_xlsx(datasets, algorithms, phase3_results, phase4_results,
                        timing, experiment_name, s3_bucket=None, phase4b_results=None,
                        phase4c_results=None, phase4d_results=None, phase4e_results=None,
                        phase4f_results=None, phase4g_results=None, phase4h_results=None, phase3b_results=None,
                        local_data_dir=None, gnn_models=None):
    """Generate plots and save unified XLSX to S3 or local disk."""
    import tempfile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import shutil

    if gnn_models is None:
        gnn_models = ['sage', 'gat', 'transformer', 'clusterscl']

    work_dir = tempfile.mkdtemp()
    sns.set_theme(style='whitegrid', palette='muted')

    for dataset in datasets:
        # Accuracy distribution
        fig, axes = plt.subplots(1, max(1, len(algorithms)), figsize=(6*max(1, len(algorithms)), 4),
                                 squeeze=False)
        for ax, alg in zip(axes[0], algorithms if algorithms else ['dummy']):
            if alg == 'dummy':
                continue
            df = None
            for m_type in gnn_models:
                df = phase3_results.get((dataset, alg, m_type))
                if df is not None:
                    break
            if df is None:
                df = phase3_results.get((dataset, alg))
            if df is not None and 'comm_test_acc' in df.columns:
                ax.hist(df['comm_test_acc'].dropna(), bins=40, edgecolor='k', alpha=0.7)
            ax.set_title(f'{dataset} / {alg}')
            ax.set_xlabel('Per-Community Test Accuracy')
            ax.set_ylabel('Count')
        plt.suptitle('Community Accuracy Distribution', fontweight='bold')
        plt.tight_layout()
        p1 = os.path.join(work_dir, f'{dataset}_acc_dist.png')
        plt.savefig(p1, dpi=150, bbox_inches='tight')
        plt.close()

        # Boundary vs Internal accuracy
        bnd_means = []
        int_means = []
        labels    = []
        for alg in algorithms:
            df = None
            for m_type in gnn_models:
                df = phase3_results.get((dataset, alg, m_type))
                if df is not None:
                    break
            if df is None:
                df = phase3_results.get((dataset, alg))
            if df is not None:
                bnd_means.append(df[df['n_boundary']>0]['boundary_acc'].mean())
                int_means.append(df[df['n_internal']>0]['internal_acc'].mean())
                labels.append(alg)
        if labels:
            x = range(len(labels))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar([i-.2 for i in x], bnd_means, .4, label='Boundary', color='coral')
            ax.bar([i+.2 for i in x], int_means, .4, label='Internal', color='steelblue')
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels)
            ax.set_ylabel('Test Accuracy')
            ax.legend()
            ax.set_title(f'{dataset}: Boundary vs Internal Node Accuracy',
                         fontweight='bold')
            plt.tight_layout()
            p2 = os.path.join(work_dir, f'{dataset}_boundary_internal.png')
            plt.savefig(p2, dpi=150, bbox_inches='tight')
            plt.close()

        # Timing stacked bar
        t1s = [timing.get(('phase1', dataset, a), 0) for a in algorithms]
        t2s = [timing.get(('phase2', dataset, a), 0) for a in algorithms]
        t3s = []
        for a in algorithms:
            t3_val = 0
            for m_type in gnn_models:
                t3_val += timing.get(('phase3', dataset, a, m_type), 0)
            if t3_val == 0:
                t3_val = timing.get(('phase3', dataset, a), 0)
            t3s.append(t3_val)
        if any(t1s + t2s + t3s):
            x = range(len(algorithms))
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(x, t1s, label='Phase1 (CD)')
            ax.bar(x, t2s, bottom=t1s, label='Phase2 (Partition)')
            ax.bar(x, t3s, bottom=[a+b for a,b in zip(t1s,t2s)], label='Phase3 (Train)')
            t4 = timing.get(('phase4', dataset), 0)
            if t4:
                ax.axhline(t4, color='red', linestyle='--', label=f'Baseline ({t4:.0f}s)')
            ax.set_xticks(list(x))
            ax.set_xticklabels(algorithms)
            ax.set_ylabel('Seconds')
            ax.legend()
            ax.set_title(f'{dataset}: Phase Timing Breakdown', fontweight='bold')
            plt.tight_layout()
            p3 = os.path.join(work_dir, f'{dataset}_timing.png')
            plt.savefig(p3, dpi=150, bbox_inches='tight')
            plt.close()

    # Save XLSX
    xlsx_path = os.path.join(work_dir, f'{experiment_name}_results.xlsx')
    with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as xw:
        summary_rows = []
        for dataset in datasets:
            for alg in algorithms:
                for m_type in gnn_models:
                    key = (dataset, alg, m_type)
                    df  = phase3_results.get(key)
                    if df is None and m_type == 'sage':
                        df = phase3_results.get((dataset, alg))
                    
                    bl = None
                    if m_type == 'sage':
                        bl = phase4_results.get(dataset)
                    elif m_type == 'gat':
                        bl = (phase4e_results or {}).get(dataset)
                    elif m_type == 'transformer':
                        bl = (phase4f_results or {}).get(dataset)
                    elif m_type == 'clusterscl':
                        bl = (phase4g_results or {}).get(dataset)
                    elif m_type == 'gatv2':
                        bl = (phase4h_results or {}).get(dataset)
                    b_acc = bl.get('test_acc') if bl else None
                    bl_auc = bl.get('link_auc') if bl else None
                    
                    if df is not None:
                        g_acc = df.attrs.get('weighted_comm_acc')
                        l_auc = df.attrs.get('weighted_comm_link_auc')
                        t_ph1 = timing.get(('phase1', dataset, alg))
                        t_ph2 = timing.get(('phase2', dataset, alg))
                        t_ph3 = timing.get(('phase3', dataset, alg, m_type), timing.get(('phase3', dataset, alg)))
                        t_base = timing.get(('phase4' if m_type == 'sage' else ('phase4e' if m_type == 'gat' else ('phase4h' if m_type == 'gatv2' else ('phase4f' if m_type == 'transformer' else 'phase4g'))), dataset))
                        
                        summary_rows.append({
                            'experiment':             experiment_name,
                            'dataset':                dataset,
                            'algorithm':              alg,
                            'model_type':             m_type,
                            'n_communities':          len(df),
                            'global_test_acc':        g_acc,
                            'mean_comm_acc':          df['comm_test_acc'].mean(),
                            'mean_boundary_acc':      df[df['n_boundary']>0]['boundary_acc'].mean() if len(df[df['n_boundary']>0]) > 0 else 0.0,
                            'mean_internal_acc':      df[df['n_internal']>0]['internal_acc'].mean() if len(df[df['n_internal']>0]) > 0 else 0.0,
                            'weighted_comm_acc':      g_acc,
                            'baseline_acc':           b_acc,
                            'acc_gap':                (b_acc - g_acc) if (b_acc and g_acc) else None,
                            'mean_comm_link_acc':     df['comm_link_auc'].mean() if 'comm_link_auc' in df.columns else None,
                            'weighted_comm_link_auc': l_auc,
                            'baseline_link_auc':      bl_auc,
                            'auc_gap':                (bl_auc - l_auc) if (bl_auc is not None and l_auc is not None) else None,
                            'phase1_s':          t_ph1,
                            'phase2_s':          t_ph2,
                            'phase3_s':          t_ph3,
                            'total_123_s':        (t_ph1 or 0.0) + (t_ph2 or 0.0) + (t_ph3 or 0.0),
                            'baseline_s':        t_base,
                        })
                    
                    if phase3b_results is not None:
                        dfb = phase3b_results.get(key)
                        if dfb is not None:
                            gb_acc = dfb.attrs.get('weighted_comm_acc')
                            lb_auc = dfb.attrs.get('weighted_comm_link_auc')
                            t_ph1 = timing.get(('phase1', dataset, alg))
                            t_ph2 = timing.get(('phase2', dataset, alg))
                            t_ph3b = timing.get(('phase3b', dataset, alg, m_type))
                            t_base = timing.get(('phase4' if m_type == 'sage' else ('phase4e' if m_type == 'gat' else ('phase4h' if m_type == 'gatv2' else ('phase4f' if m_type == 'transformer' else 'phase4g'))), dataset))
                            
                            summary_rows.append({
                                'experiment':             experiment_name,
                                'dataset':                dataset,
                                'algorithm':              alg,
                                'model_type':             f"{m_type}-caan",
                                'n_communities':          len(dfb),
                                'global_test_acc':        gb_acc,
                                'mean_comm_acc':          dfb['comm_test_acc'].mean(),
                                'mean_boundary_acc':      dfb[dfb['n_boundary']>0]['boundary_acc'].mean() if len(dfb[dfb['n_boundary']>0]) > 0 else 0.0,
                                'mean_internal_acc':      dfb[dfb['n_internal']>0]['internal_acc'].mean() if len(dfb[dfb['n_internal']>0]) > 0 else 0.0,
                                'weighted_comm_acc':      gb_acc,
                                'baseline_acc':           b_acc,
                                'acc_gap':                (b_acc - gb_acc) if (b_acc and gb_acc) else None,
                                'mean_comm_link_acc':     dfb['comm_link_auc'].mean() if 'comm_link_auc' in dfb.columns else None,
                                'weighted_comm_link_auc': lb_auc,
                                'baseline_link_auc':      bl_auc,
                                'auc_gap':                (bl_auc - lb_auc) if (bl_auc is not None and lb_auc is not None) else None,
                                'phase1_s':          t_ph1,
                                'phase2_s':          t_ph2,
                                'phase3_s':          t_ph3b,
                                'total_123_s':        (t_ph1 or 0.0) + (t_ph2 or 0.0) + (t_ph3b or 0.0),
                                'baseline_s':        t_base,
                            })
        pd.DataFrame(summary_rows).to_excel(xw, index=False, sheet_name='summary')

        for dataset in datasets:
            for alg in algorithms:
                for m_type in gnn_models:
                    key = (dataset, alg, m_type)
                    df = phase3_results.get(key)
                    if df is None and m_type == 'sage':
                        df = phase3_results.get((dataset, alg))
                    if df is not None:
                        sheet = f'{dataset[:8]}_{alg}_{m_type}'[:31]
                        df.drop(columns=['test_preds'], errors='ignore')\
                          .to_excel(xw, index=False, sheet_name=sheet)
                    
                    if phase3b_results is not None:
                        dfb = phase3b_results.get(key)
                        if dfb is not None:
                            sheetb = f'caan_{dataset[:5]}_{alg}_{m_type}'[:31]
                            dfb.drop(columns=['test_preds'], errors='ignore')\
                              .to_excel(xw, index=False, sheet_name=sheetb)

        bl_rows = [{'dataset': ds, **v} for ds, v in phase4_results.items()]
        if bl_rows:
            pd.DataFrame(bl_rows).to_excel(xw, index=False, sheet_name='phase4_baseline')

        if phase4b_results:
            b4b_rows = [{'dataset': ds, **v} for ds, v in phase4b_results.items()]
            if b4b_rows:
                pd.DataFrame(b4b_rows).to_excel(xw, index=False, sheet_name='phase4b_distdgl')

        if phase4c_results:
            b4c_rows = [{'dataset': ds, **v} for ds, v in phase4c_results.items()]
            if b4c_rows:
                pd.DataFrame(b4c_rows).to_excel(xw, index=False, sheet_name='phase4c_arma')

        if phase4d_results:
            b4d_rows = [{'dataset': ds, **v} for ds, v in phase4d_results.items()]
            if b4d_rows:
                pd.DataFrame(b4d_rows).to_excel(xw, index=False, sheet_name='phase4d_asap')

        if phase4e_results:
            b4e_rows = [{'dataset': ds, **v} for ds, v in phase4e_results.items()]
            if b4e_rows:
                pd.DataFrame(b4e_rows).to_excel(xw, index=False, sheet_name='phase4e_gat')

        if phase4f_results:
            b4f_rows = [{'dataset': ds, **v} for ds, v in phase4f_results.items()]
            if b4f_rows:
                pd.DataFrame(b4f_rows).to_excel(xw, index=False, sheet_name='phase4f_trans')

        if phase4g_results:
            b4g_rows = [{'dataset': ds, **v} for ds, v in phase4g_results.items()]
            if b4g_rows:
                pd.DataFrame(b4g_rows).to_excel(xw, index=False, sheet_name='phase4g_clusterscl')

        if phase4h_results:
            b4h_rows = [{'dataset': ds, **v} for ds, v in phase4h_results.items()]
            if b4h_rows:
                pd.DataFrame(b4h_rows).to_excel(xw, index=False, sheet_name='phase4h_gatv2')

    # Save to S3 or local directory
    if local_data_dir is not None:
        target_xlsx = os.path.join(local_data_dir, 'gnn-bench-out', f'{experiment_name}_results.xlsx')
        os.makedirs(os.path.dirname(target_xlsx), exist_ok=True)
        shutil.copyfile(xlsx_path, target_xlsx)
        print(f"  XLSX → {target_xlsx}")
        
        plot_dir = os.path.join(local_data_dir, 'gnn-bench-out', 'plots', experiment_name)
        os.makedirs(plot_dir, exist_ok=True)
        for fname in os.listdir(work_dir):
            if fname.endswith('.png'):
                shutil.copyfile(os.path.join(work_dir, fname), os.path.join(plot_dir, fname))
                print(f"  Plot → {os.path.join(plot_dir, fname)}")
    else:
        import boto3
        s3 = boto3.client('s3')
        s3.upload_file(xlsx_path, s3_bucket,
                       f'gnn-bench-out/{experiment_name}_results.xlsx')
        print(f"  XLSX → s3://{s3_bucket}/gnn-bench-out/{experiment_name}_results.xlsx")
        for fname in os.listdir(work_dir):
            if fname.endswith('.png'):
                lp  = os.path.join(work_dir, fname)
                key = f'gnn-bench-out/plots/{experiment_name}/{fname}'
                s3.upload_file(lp, s3_bucket, key)
                print(f"  Plot → s3://{s3_bucket}/{key}")
                
    shutil.rmtree(work_dir, ignore_errors=True)


def print_summary(experiment_name, datasets, algorithms, use_global_mapping,
                  min_size, phase1_results, phase2_results, phase3_results,
                  phase4_results, timing, phase4b_results=None,
                  phase4c_results=None, phase4d_results=None, phase4e_results=None,
                  phase4f_results=None, phase4g_results=None, phase4h_results=None, phase3b_results=None, gnn_models=None):
    """Print the final experiment summary box."""
    W = 62
    def row(label, val, width=W):
        line = f"  {label:<30} {val}"
        return f"║ {line:<{width}} ║"

    if gnn_models is None:
        gnn_models = ['sage', 'gat', 'gatv2', 'transformer', 'clusterscl']

    print("╔" + "═"*W + "╗")
    print(f"║  EXPERIMENT: {experiment_name:<{W-15}}║")
    print(f"║  Global mapping: {str(use_global_mapping):<{W-18}}║")
    print("╠" + "═"*W + "╣")

    for dataset in datasets:
        print(row("Dataset", dataset))
        if ('phase0', dataset) in timing:
            print(row("Phase 0 (ingest)", f"{timing[('phase0',dataset)]:.1f}s"))
        else:
            print(row("Phase 0 (ingest)", "SKIPPED"))
        for alg in algorithms:
            t1 = timing.get(('phase1', dataset, alg))
            t2 = timing.get(('phase2', dataset, alg))
            if t1: print(row(f"Phase 1 [{alg}]", f"{t1:.1f}s"))
            if t2: print(row(f"Phase 2 [{alg}]", f"{t2:.1f}s"))
            for m_type in gnn_models:
                t3 = timing.get(('phase3', dataset, alg, m_type))
                if t3 is None and m_type == 'sage':
                    t3 = timing.get(('phase3', dataset, alg))
                if t3:
                    print(row(f"Phase 3 [{alg}-{m_type}]", f"{t3:.1f}s"))
                t3b = timing.get(('phase3b', dataset, alg, m_type))
                if t3b:
                    print(row(f"Phase 3b [{alg}-{m_type}-caan]", f"{t3b:.1f}s"))
        t4 = timing.get(('phase4', dataset))
        if t4: print(row("Phase 4 (baseline)", f"{t4:.1f}s"))
        t4b = timing.get(('phase4b', dataset))
        if t4b: print(row("Phase 4b (DistDGL)", f"{t4b:.1f}s"))
        t4c = timing.get(('phase4c', dataset))
        if t4c: print(row("Phase 4c (ARMA)", f"{t4c:.1f}s"))
        t4d = timing.get(('phase4d', dataset))
        if t4d: print(row("Phase 4d (ASAP)", f"{t4d:.1f}s"))
        t4e = timing.get(('phase4e', dataset))
        if t4e: print(row("Phase 4e (GAT)", f"{t4e:.1f}s"))
        t4f = timing.get(('phase4f', dataset))
        if t4f: print(row("Phase 4f (GraphTrans)", f"{t4f:.1f}s"))
        t4g = timing.get(('phase4g', dataset))
        if t4g: print(row("Phase 4g (ClusterSCL)", f"{t4g:.1f}s"))
        t4h = timing.get(('phase4h', dataset))
        if t4h: print(row("Phase 4h (GATv2)", f"{t4h:.1f}s"))
        
        bl  = phase4_results.get(dataset)
        print("╠" + "═"*W + "╣")
        for alg in algorithms:
            for m_type in gnn_models:
                df  = phase3_results.get((dataset, alg, m_type))
                if df is None and m_type == 'sage':
                    df = phase3_results.get((dataset, alg))
                if df is not None:
                    g   = df.attrs.get('weighted_comm_acc', float('nan'))
                    b   = df[df['n_boundary']>0]['boundary_acc'].mean() if len(df[df['n_boundary']>0]) > 0 else 0.0
                    i   = df[df['n_internal']>0]['internal_acc'].mean() if len(df[df['n_internal']>0]) > 0 else 0.0
                    l_auc = df.attrs.get('weighted_comm_link_auc', float('nan'))
                    print(row(f"[{alg}-{m_type}] global acc", f"{g:.4f}"))
                    print(row(f"[{alg}-{m_type}] boundary acc", f"{b:.4f}"))
                    print(row(f"[{alg}-{m_type}] internal acc", f"{i:.4f}"))
                    print(row(f"[{alg}-{m_type}] link AUC", f"{l_auc:.4f}"))
                
                # CaaN Global Graph Results
                if phase3b_results is not None:
                    dfb = phase3b_results.get((dataset, alg, m_type))
                    if dfb is not None:
                        gb = dfb.attrs.get('weighted_comm_acc', float('nan'))
                        bb = dfb[dfb['n_boundary']>0]['boundary_acc'].mean() if len(dfb[dfb['n_boundary']>0]) > 0 else 0.0
                        ib = dfb[dfb['n_internal']>0]['internal_acc'].mean() if len(dfb[dfb['n_internal']>0]) > 0 else 0.0
                        lb_auc = dfb.attrs.get('weighted_comm_link_auc', float('nan'))
                        print(row(f"[{alg}-{m_type}-caan] global acc", f"{gb:.4f}"))
                        print(row(f"[{alg}-{m_type}-caan] boundary acc", f"{bb:.4f}"))
                        print(row(f"[{alg}-{m_type}-caan] internal acc", f"{ib:.4f}"))
                        print(row(f"[{alg}-{m_type}-caan] link AUC", f"{lb_auc:.4f}"))
                        
            r2   = phase2_results.get((dataset, alg), {})
            _fmt = lambda v: f'{v:,}' if isinstance(v, int) else str(v)
            print(row(f"[{alg}] communities",
                      f"{_fmt(r2.get('n_comms_raw','?'))} raw → "
                      f"{_fmt(r2.get('n_valid_comms','?'))} valid"))
        if bl:
            bl_acc = bl.get('test_acc', 0.0)
            bl_auc = bl.get('link_auc', 0.0)
            print(row("Baseline acc", f"{bl_acc:.4f}"))
            print(row("Baseline link AUC", f"{bl_auc:.4f}  ({bl['train_time_s']:.1f}s)"))
        if phase4b_results:
            bl4b = phase4b_results.get(dataset)
            if bl4b:
                bl4b_acc = bl4b.get('test_acc', 0.0)
                bl4b_auc = bl4b.get('link_auc', 0.0)
                print(row("DistDGL acc", f"{bl4b_acc:.4f}"))
                print(row("DistDGL link AUC", f"{bl4b_auc:.4f}  ({bl4b['train_time_s']:.1f}s)"))
        if phase4c_results:
            bl4c = phase4c_results.get(dataset)
            if bl4c:
                bl4c_acc = bl4c.get('test_acc', 0.0)
                bl4c_auc = bl4c.get('link_auc', 0.0)
                print(row("ARMA acc", f"{bl4c_acc:.4f}"))
                print(row("ARMA link AUC", f"{bl4c_auc:.4f}  ({bl4c['train_time_s']:.1f}s)"))
        if phase4d_results:
            bl4d = phase4d_results.get(dataset)
            if bl4d:
                bl4d_acc = bl4d.get('test_acc', 0.0)
                bl4d_auc = bl4d.get('link_auc', 0.0)
                print(row("ASAP acc", f"{bl4d_acc:.4f}"))
                print(row("ASAP link AUC", f"{bl4d_auc:.4f}  ({bl4d['train_time_s']:.1f}s)"))
        if phase4e_results:
            bl4e = phase4e_results.get(dataset)
            if bl4e:
                bl4e_acc = bl4e.get('test_acc', 0.0)
                bl4e_auc = bl4e.get('link_auc', 0.0)
                print(row("GAT acc", f"{bl4e_acc:.4f}"))
                print(row("GAT link AUC", f"{bl4e_auc:.4f}  ({bl4e['train_time_s']:.1f}s)"))
        if phase4f_results:
            bl4f = phase4f_results.get(dataset)
            if bl4f:
                bl4f_acc = bl4f.get('test_acc', 0.0)
                bl4f_auc = bl4f.get('link_auc', 0.0)
                print(row("GraphTrans acc", f"{bl4f_acc:.4f}"))
                print(row("GraphTrans link AUC", f"{bl4f_auc:.4f}  ({bl4f['train_time_s']:.1f}s)"))
        if phase4g_results:
            bl4g = phase4g_results.get(dataset)
            if bl4g:
                print(row("ClusterSCL acc", f"{bl4g['test_acc']:.4f}"))
                print(row("ClusterSCL link AUC", f"{bl4g['link_auc']:.4f}  ({bl4g['train_time_s']:.1f}s)"))
        if phase4h_results:
            bl4h = phase4h_results.get(dataset)
            if bl4h:
                print(row("GATv2 acc", f"{bl4h['test_acc']:.4f}"))
                print(row("GATv2 link AUC", f"{bl4h['link_auc']:.4f}  ({bl4h['train_time_s']:.1f}s)"))

    print("╚" + "═"*W + "╝")
