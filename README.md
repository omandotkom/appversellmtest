# Gateway Model Lab

Dashboard monitoring + benchmark untuk **banyak provider** LLM (OpenAI-compatible: LiteLLM,
OpenRouter, vLLM, Ollama, dan sejenisnya). Setiap model diidentifikasi sebagai
`(provider_id, model_name)`, jadi nama model yang sama di dua provider tetap dibedakan saat
pengujian, perbandingan, dan histori.

## Fitur

- **Katalog multi-provider**: tambah/edit provider, simpan API key + admin key di SQLite,
  test koneksi, dan refresh daftar model per provider.
- **Tes fungsional** (murah, cocok berkala): chat dasar, streaming + TTFT, tool calling, JSON mode.
- **Comparison lintas provider/model**: satu tombol menjalankan 4 cek fungsional penuh dengan
  multi-trial + warmup, lalu memberi skor 0–100 tertimbang (tok/s, TTFT, pass rate, ITL,
  latency, biaya). Tersedia juga mode histori `latest` / `median` per window.
- **Task benchmark** (coding-agent): model mengerjakan task nyata (mis. buat landing page) lewat
  loop multi-turn streaming dengan tools `read_file`, `write_file`, `edit_file`, `list_files`,
  `grep`. Workspace tiap target terisolasi.
- **Metrik error per tool call** + kategori, recovery rate, outcome, turn, token, estimasi biaya.
- **Performance suite**: load/concurrency probe (throughput + 429) dan context scaling
  (prefill tok/s), plus deteksi regresi per target.
- **Trace turn-by-turn** dan preview artefak hasil (iframe sandbox).
- **History** run di SQLite, UI single-page tanpa build step.

## Instalasi

```powershell
cd C:\SM_WORKSPACE\appversellmtest
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt -r requirements-dev.txt
```

Alternatif tanpa `uv`:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r requirements-dev.txt
```

## Menjalankan

```powershell
.venv\Scripts\python run.py
```

Buka http://127.0.0.1:8000, masuk ke tab **Providers**, lalu tambahkan provider:

- **Base URL** contoh `http://localhost:4000` atau `https://openrouter.ai/api`
- **API key** dipakai untuk request chat (jalur pelanggan)
- **Admin key** opsional untuk endpoint metadata model LiteLLM (`/v2/model/info`)
- **Chat path** default `/v1/chat/completions`; **models path** default `/v1/models`

Setelah disimpan, klik **Test** lalu **Refresh** pada provider untuk mengisi katalog model.
Test melakukan discovery metadata **dan** satu chat kecil memakai API key (jalur pelanggan), sehingga
key yang salah langsung terlihat. API key tidak pernah dikembalikan oleh API; UI hanya menampilkan
`has_api_key` / `has_admin_key`.

## Mode demo (tanpa token)

Mock gateway menyediakan dua provider terpisah dalam satu proses:

```powershell
# terminal 1
.venv\Scripts\python mock\mock_gateway.py

# terminal 2
.venv\Scripts\python run.py
```

Di tab Providers tambahkan:

| Nama | Base URL |
|---|---|
| Mock Provider A | `http://127.0.0.1:4010/a` |
| Mock Provider B | `http://127.0.0.1:4010/b` |

Keduanya punya `mock-good`, sehingga duplikasi nama model lintas provider ikut teruji.
Isi API key dummy apa saja, Test, lalu Refresh.

Verifikasi end-to-end otomatis:

```powershell
.venv\Scripts\python scripts\smoke_test.py
```

## Konfigurasi (.env)

Hanya konfigurasi global non-rahasia yang disimpan di `.env`. Key provider ada di SQLite.

| Key | Deskripsi |
|---|---|
| `HOST` / `PORT` | Alamat dashboard (default `127.0.0.1:8000`) |
| `DB_PATH` | Lokasi SQLite (default `data/health.db`) |
| `FUNCTIONAL_CONCURRENCY` | Concurrency tes fungsional (default 4) |
| `BENCH_CONCURRENCY` | Concurrency benchmark (default 2) |
| `DEFAULT_TIMEOUT_S` | Timeout per request (default 60) |
| `BENCH_TOKEN_CAP` | Batas token per model per benchmark (default 150000) |
| `LOAD_LEVELS` / `LOAD_REQUESTS_PER_LEVEL` | Default load probe |
| `CONTEXT_SIZES` | Default ukuran context scaling |
| `REGRESSION_THRESHOLD_PCT` | Ambang regresi (default 25%) |
| `COMPARISON_WINDOW` / `COMPARISON_REPEAT` | Default histori comparison |
| `SCORE_W_*` | Bobot skor comparison |

Semua nilai di atas juga dapat diubah dari tab **Settings**.

## Reset database

Schema sekarang `user_version = 2`. Saat aplikasi start, database lama (single-gateway,
`user_version < 2`) otomatis di-rename menjadi `data/health.db.bak.<timestamp>` lalu database baru
dibuat. Tidak ada migrasi data lama; histori run lama tetap ada di file backup.

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
| `provider_error` / `http_error` | kegagalan di sisi provider |

`errorRate = error tool calls / total tool calls`. Outcome: `completed`, `completed_incomplete`,
`max_turns`, `token_cap`, `request_error`, `aborted`.

## Struktur

```
app/
  main.py             # FastAPI: API + UI + orkestrasi run
  catalog.py          # CRUD provider, test koneksi, discovery model
  targets.py          # resolusi target (provider_id, model)
  providers/
    base.py           # kontrak ModelInfo / ChatResult / BaseProvider
    openai_compat.py  # adapter OpenAI-compatible (chat, stream, tools, JSON)
  comparison.py       # skor + rollup lintas provider
  runner.py           # tes fungsional (deep latency + persentil)
  perf.py             # load probe + context scaling
  db.py               # SQLite schema v2
  config.py           # .env global
  agent/
    loop.py           # agent loop multi-turn (tanpa agent SDK)
    tools.py          # tool file + sandbox path
    errors.py         # agregasi metrik error
    grading.py        # cek artefak
  tasks/              # definisi task benchmark (YAML)
  static/             # UI (HTML/CSS/JS)
mock/                 # mock gateway 2 provider (/a, /b)
scripts/              # smoke test
tests/                # test backend (pytest)
```

## Keamanan & catatan

- API key provider disimpan di SQLite. API hanya mengembalikan status `has_api_key` /
  `has_admin_key`, bukan nilainya.
- Preview artefak di-serve dengan `Content-Security-Policy: sandbox` dan iframe `sandbox=""`
  (script nonaktif).
- Workspace tiap target terisolasi di `data/bench/{run_id}/p{provider_id}/{model}/`; tool file
  menolak akses keluar workspace (`..` dan path absolut diblokir).
- Provider dihapus secara soft (`enabled=0`) agar histori run tetap terbaca.

## Roadmap

- Adapter native Anthropic dan Gemini.
- `run_command` (bash) untuk dimensi "run tests → fix" — via WSL/Docker/cloud sandbox.
- Penilaian tambahan LLM judge (opsional, non-estetika).
- Notifikasi webhook saat ada model gagal.