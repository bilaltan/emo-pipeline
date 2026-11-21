# Table A results

One folder per run set, named `<split>_c<C_max>`. Each has its own README (setup, caveats, commits), the raw
`write.jsonl` / `read.jsonl` / `halo*.jsonl` records, and `tableA.md` + `tableA.csv`. The CSVs share their columns,
so two run sets compare by joining on (dataset, layout). Written tables live in WSL under
`~/lakegrl/tableA/<run set>/`.

| Run set | Split of units above C_max | C_max | Dates | Status |
| --- | --- | --- | --- | --- |
| `idmod_c10000/` | Stage 3b `id mod n` blocks (ablation baseline) | 10,000 | 2026-10-27 to 10-30 | final (Table A + halo sub-table) |
| `rleiden_c10000/` | Recursive Leiden to <= C_max/2 core members (Phase 1), halo <= core (Phase 2) | 10,000 | from 2026-10-30 | running |

Later: the C_max sensitivity runs (5,000, 20,000, 40,000) go in `rleiden_c<C_max>/`.
