# Changelog

One line per change. The newest heading is the engine's version (shown on the server's banner and at `/version`).

## 0.5.0
- copy drafts P168 (`[decode] copy`, `copy_rows` 64, `copy_match` 8; server/copy_draft.py): a request decoding alone whose reply repeats text of its history verifies the tokens that followed that text there (first copy 7 tokens, doubling while taken whole, up to 63) in place of DSpark's drafts, each row drawn as plain decoding draws it — a 1,591-token file rewritten: 92.9 → 311 tok/s, 5.4 → 39 tokens a round, tokens identical to copy off; kept rows bitwise one-row decoding's at 16 / 32 / 64 rows, also at 140K context (wide_check); logit rows 64 (+~96 MiB a rank)
- DSpark draft pass, bitwise: the sampled Markov sweep backwards on odd steps P161, q halves gathered into rows in place P162 — round B=1 54.65 → 53.98 ms, B=4 91.94 → 91.61 (P160 context rows on a side stream and P163 split sweep measured, off)
- decode: the HCA compressor back inline (P074 hca_side off; same kernels, same bits) — 1 row 31.3 → 30.9 ms, verify 5 / 16 rows 49.7 / 78.1 → 49.3 / 77.8
- decode / verify kernels, bitwise: mHC at M ≤ 32 in one wave with register-blocked partials P164 (M 1 / 16 / 32: 14.3 / 20.3 / 26.9 → 9.0 / 15.0 / 19.1 µs a call), router / compressor at M 17–32 as 64 two-warp CTAs P165
- prefill, bitwise: P100's o exchange put straight into the peer's registered receive buffer P176 (+129 MiB a rank) — 8192-row chunk at 0 2,111 → 2,094 ms (3,912 tok/s); long-context HCA attention with the K-tile gather rescheduled, from 5 key chunks P172 — chunk at 131K 3,027 → 2,850 ms, at 524K 5,247 → 5,185
- measured, not built: decode verify step's small parts (knock-outs: 8.1 ms at 5 rows vs ~4.4 floor, each gap a redesign), HCA long-context attention ‖ MoE on one GPU (2.2 % / 0.3 % — no lane overlap), CUDA graphs (no launch gap), a warp-specialized HCA long-context attention P180 (setmaxnreg, bitwise; 524K 15.72 → 16.52 ms a call, 1M 39.35 → 38.97 — off: the 2-buffer K ring in 99 KB of shared memory and the exact FP8 → bf16 K fill bound it)
- serving (./start.sh, 0.5.0): server_check smoke / tools / thinking pass; a 1,655-token file rewrite over HTTP at the default sampling 306.7 tok/s, the file exact; bench_lite decode 1 at once (8K, thinking) 54.7 → 57.3 tok/s; start 204 s with the one-time engine build (140 s), ~1 min without
- server_check: the streamed tool call passes with an empty thinking (at T 1 ERNIE may close it at once — a plain draw of the first token)
- tools: wide_check.py, sched_copy.py, round_prof.py, test_markov_back.py, test_overlap.py, zcmem.py; scheduler round cost ROUND_REST_MS 5.9 / 11.0

## 0.4.0
- prefill, bitwise equal to 0.3.0 (patch_check: 16K lanes, chunks, decode, multi-row, DSpark drafts): FP8 split chains in one CTA P133 + the shared expert's SiTU in its gate/up store P134, the shared expert sequence-parallel P136 (+504 MiB a rank), the DSA indexer's selection split over the ranks P140–141, P091 also above 65,536 candidates P142 — 8192-row chunk 2,183 → 2,113 ms at 0 (3,878 tok/s), 2,303 → 2,199 ms at 16K (3,725), 3,049 ms at 131K (2,686); decode at 9.4K 32.0 → 31.5 ms
- prefix checkpoints P146 (server.checkpoints / checkpoint_every): a prompt that leaves a slot's history part-way restores the longest checkpoint below the common prefix — a 33K turn without the last reply's reasoning: first token 9.49 → 0.19 s; bitwise a full prefill's
- DSpark: the round's verify lengths chosen together by the measured round cost P143 ([decode] verify = "cost"; "gate" = P121 alone); multi-slot rounds, penalties and KvGuard waits tested on the GPU (tokens = plain decoding's)
- serving: context 1,048,576 (server.context), a request without max_tokens gets server.max_tokens = 131,072; server_check's needle is one request with three facts (10 / 50 / 90 %) per length, exact / reuse only when named
- long context, bitwise equal (patch_check vs the 0.4.0 default): HCA prefill attention with ≥ 6 key chunks in items handed to persistent CTAs, partials merged from L2 P150; the indexer select for rows of ≥ 131,072 candidates with an exact lower bound and compact kept keys, no fp32 logits round trip P153 — 8192-row chunk at 524K ≈ 7.8 → 5.25 s; a cold 995K prompt 625.8 s (1,591 tok/s; 998,950 tokens took 960 s before), 128K 40.2 s (3,264), 512K 235 s (2,231); 1M needle ✓✓✓
- server.prefill_marks (measurement, off): each prefill chunk waited for and one [prefill] line a request — the time to first token of a cold 8K / 32K / 128K / 256K / 512K / full prompt from one long read; bench_lite --streams (decode 1 … 4 at once) and long prompts fitted by /tokenize
- tools: sched_ckpt.py, sched_dspark.py multi / cost / rate / kvwait, xfer_prof.py, xfer_trace.py (E55_COMM_TRACE), test_dsa --split / --wide, test_gemm sites at the 8192-row chunk

## 0.3.0
- prefill, bitwise equal to 0.2.1 (patch_check: 16K lanes, chunks, decode, multi-row, DSpark drafts): attention output side on the row owner by one o all-to-all + its schedule + late shared experts P100–102, FP8 prefill tiles / epilogues / chains P103–105, mHC prefill P106–107, DSA attention epilogue P110, large RDMA calls by the copy engines P131 (P132 NUMA staging: off) — 8192-row chunk 2,183 ms at 0 (3,752 tok/s; 0.2.1 2 × 1,304 ms), 2,303 ms at 16K
- decode / verify, bitwise: router / compressor halves in one launch + small-grid linears P114–115, RDMA small calls (BlueFlame posts, host sequence, small-call plan) P118–120, DSpark draft attention / Markov chain P129–130 — decode at 9.4K 32.5 → 32.0 ms, verify 4 / 8 rows 44.4 / 58.0 → 43.3 / 57.1 ms
- the server decodes by DSpark rounds ([decode] dspark, block 7, gate 0.3; logit rows slots × (block + 1)): per-round caps at max_tokens / thinking budget / loop guard / end-token drafts, confidence gate P121, verify ≤ cap − 1 drafts P122 — sched_dspark T = 0 thinking + code: tokens identical to plain greedy, kept rows' logits bitwise (447/447, 319/319)
- KV pool P125: the growing caches from one pool of 2 MiB chunks, sized at start after a warm-up of the serving paths (min MemFree − memory_free_gib, kv_pool_gib cap), requests admitted by the pool (output_reserve, memory_floor_gib watch, /health · /metrics), up to 1,048,576 tokens a request — 4 slots × 64K, chunk 8192, DSpark: engine 83.1 + draft 6.1 GiB a rank, pool 6,228 chunks = 12.16 GiB ≈ 2.89M tokens, MemFree after 15.00 / 15.16 GiB (reserve 15); bitwise (patch_check)
- tools: sched_dspark.py, kv_admit.py, test_bigxfer.py, test_comm.py probes, step_prof --lanes-env

## 0.2.1
- prefill, bitwise equal to 0.2.0 (patch_check: 16K lanes, chunks, decode, multi-row, DSpark drafts): sequence-parallel residual P094, HCA attention / pre P088–89, DSA select / attention / bmm P091–93, MoE combine P086, router / compressor from the 12-bit copy P098 — 4096-row chunk 1,562 → 1,398 ms at 0, 1,666 → 1,454 ms at 16K
- memory: embedding as vocabulary halves P095, lm_head only as the 12-bit copy P097, router / compressor only as the 12-bit copy P098 (≈ 2.2 GiB per rank); the engine frees everything on close; the server's memory check counts free pages and the measured engine size
- serve.toml chunk 8192 (was 4096): 4096 rows of prefill 1,398 → 1,304 ms at 0 (3,140 tok/s), the engine ~89 GiB a rank at 4 × 64K with DSpark
- tools: step_prof --verify / --lanes, mem_report.py, test_lin12.py; tp2.sh runs one at a time (Mac lock)

## 0.2.0
- numerics aligned to Baidu's serving (vLLM-ERNIE TP2): MoE bf16 combine, mHC decode-path mixer for every M, vLLM SiTU, routing slot order + tree sum, DSA / indexer bf16 cos/sin
- batch-1 kernels, bitwise equal to the base: FP8 GEMM P030–34, MoE P035–37, HCA P040–44, DSA P045–48, mHC P050–51
- batch-2: fused ops around FP8 GEMMs + PDL + 12-bit router/compressor + sparse logits P060–64, MoE combine / fused decode P065–66, attention / mHC P070–74, prefill lanes with async all-reduces + RDMA gather + split replicated weights + larger staging P075–79
- DSpark speculative decoding (engine + binding API; draft patches P080–83), lossless (tokens and kept-row logits equal plain decode)
- engine binding calls run without the GIL
- gate taps the mid-layer mHC state (exact isolated-layer checks); engine/tests/patch_check.py (all patches off vs on, bitwise)
- development tools: tools/agent.sh sandboxes with a GPU lock; tp2.sh E55_GPU_LOCK / E55_SYNC_FILES / E55_TREE

## 0.1.0
- first version
