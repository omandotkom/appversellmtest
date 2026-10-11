# Gateway Model Lab

Dashboard monitoring + benchmark untuk AI gateway (LiteLLM-compatible, seperti OpenRouter).
Menguji model-model yang terdaftar lewat jalur yang persis dialami pelanggan Anda:
dari tes fungsional cepat sampai benchmark tugas coding-agent dengan metrik error per tool call.

## Fitur

- **Tes fungsional** (murah, cocok berkala): chat dasar, streaming + TTFT, tool calling, JSON mode.
- **Task benchmark** (coding-agent): model mengerjakan task nyata (mis. buat landing page) lewat
  loop multi-turn (streaming) dengan tools `read_file`, `write_file`, `edit_file`, `list_files`, `grep`.
- **Metrik error per tool call** + breakdown per kategori dan per tool, recovery rate, outcome,
  jumlah turn, token, estimasi biaya, durasi.
- **Performance suite**: tok/s, ITL, TTFT vs generation time, p50/p95 (multi-trial + warmup),
  load/concurrency probe (throughput + 429), context scaling (prefill tok/s), deteksi regresi.
- **Speed Comparison**: tab chart komparasi performa antar model (Chart.js) — tok/s fungsional,
  TTFT p50/p95, ITL, throughput load probe, prefill tok/s per ukuran context, tok/s benchmark —
  plus overall score 0–100 tertimbang, diagregasi dari seluruh run historis di SQLite.
- **Trace turn-by-turn** dan preview artefak hasil (iframe sandbox).
- **History** run, simpan di SQLite.
- UI single-page tanpa build step (FastAPI + vanilla JS).

## Instalasi

```powershell
cd E:\WORKSPACE\coding-agent-test
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

## Menjalankan

```powershell
.venv\Scripts\python run.py
```

Buka http://127.0.0.1:8000 lalu isi koneksi gateway di tab **Settings** (URL + admin key + virtual
test key). Key disimpan di file `.env` lokal dan tidak pernah ditampilkan kembali.

## Mode demo (tanpa token)

Ada mock gateway untuk mencoba seluruh fitur tanpa biaya:

```powershell
# terminal 1
.venv\Scripts\python mock\mock_gateway.py

# terminal 2 (setelah set GATEWAY_URL=http://127.0.0.1:4010 dan GATEWAY_TEST_KEY=sk-dummy di Settings)
.venv\Scripts\python run.py
```

Lalu jalankan `python scripts/smoke_test.py` untuk verifikasi end-to-end otomatis.

## Konfigurasi (.env)

| Key | Deskripsi |
|---|---|
| `GATEWAY_URL` | Base URL gateway (default `http://localhost:4000`) |
| `GATEWAY_ADMIN_KEY` | Admin key untuk daftar model + metadata (opsional) |
| `GATEWAY_TEST_KEY` | Virtual key untuk tes fungsional & benchmark (jalur pelanggan) |
| `FUNCTIONAL_CONCURRENCY` | Concurrency tes fungsional (default 4) |
| `BENCH_CONCURRENCY` | Concurrency benchmark (default 2) |
| `DEFAULT_TIMEOUT_S` | Timeout per request (default 60) |
| `BENCH_TOKEN_CAP` | Batas token per model per benchmark (default 150000) |

## Menambah task benchmark

Buat file YAML baru di `app/tasks/`, contoh:

```yaml
id: buat-todo-app
name: "Todo App"
prompt: |
  Buat file index.html berisi aplikasi todo sederhana ...
tools: [read_file, write_file, edit_file, list_files, grep]
max_turns: 40
artifacts:
  - path: index.html
    min_bytes: 1200
```

`artifacts` adalah penanda progres minimal (bukan penilaian estetika). Outcome ditentukan dari
apakah agent menyelesaikan task dan apakah artefak minimal terbentuk.

## Taksonomi error per tool call

| Kategori | Makna |
|---|---|
| `invalid_json_args` | arguments dari model bukan JSON valid |
| `unknown_tool` | model memanggil tool yang tidak ada |
| `schema_violation` | parameter kurang / tipe salah |
| `path_escape` | akses file di luar workspace (diblokir) |
| `file_not_found` | baca/edit file yang tidak ada |
| `edit_not_found` | `old_string` tidak ditemukan |
| `edit_ambiguous` | `old_string` cocok lebih dari satu kali |
| `tool_execution_error` | tool gagal mengeksekusi |
| `gateway_error` / `http_error` | kegagalan di sisi gateway/provider |

`errorRate = error tool calls / total tool calls`. Outcome: `completed`, `completed_incomplete`,
`max_turns`, `token_cap`, `request_error`, `aborted`.

## Struktur

```
app/
  main.py           # FastAPI: API + UI + orkestrasi run
  gateway.py        # klien gateway (list model + chat completion)
  runner.py         # tes fungsional (deep latency + persentil)
  perf.py           # load probe + context scaling
  db.py             # SQLite
  config.py         # .env
  agent/
    loop.py         # agent loop multi-turn (tanpa agent SDK)
    tools.py        # tool file + sandbox path
    errors.py       # agregasi metrik error
    grading.py      # cek artefak
  tasks/            # definisi task benchmark (YAML)
  static/           # UI (HTML/CSS/JS)
mock/               # mock gateway untuk demo/dev
scripts/            # smoke test
```

## Keamanan & catatan

- Keys hanya disimpan di `.env` lokal; UI menampilkan status "tersimpan" tanpa isinya.
- Preview artefak di-serve dengan `Content-Security-Policy: sandbox` dan iframe `sandbox=""`
  (script nonaktif).
- Workspace tiap model terisolasi di `data/bench/{run_id}/{model}/`; file tools menolak akses
  keluar workspace (`..` dan path absolut diblokir).

## Roadmap

- `run_command` (bash) untuk dimensi "run tests → fix" — via WSL/Docker/cloud sandbox.
- Penilaian tambahan LLM judge (opsional, non-estetika).
- Notifikasi webhook saat ada model gagal.
