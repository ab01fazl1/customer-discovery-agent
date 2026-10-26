# Customer Discovery Agent

Four-stage funnel that finds purchase-intent messages in a chat stream.
Stages are independent of where messages come from.

```
MessageSource --> [1 prefilter] --> [2 embedding] --> [3 local LLM] --> [4 Claude fallback] --> leads + SQLite + cost
 (CSV / JSONL /       regex          similarity        JSON scoring       ambiguous only
  Telegram)           free           ~free             local, free        $ per call
```

## Try it (no Telegram, no models, no network)

```bash
python -m discovery.synthetic --days 3 --per-day 1000 --out data/synthetic.csv
python -m discovery --csv data/synthetic.csv --db "" --no-cloud
python -m unittest discover -s tests
```

The `offline` backend uses a hashing embedder and a regex scorer so the whole
funnel runs anywhere. The CSV has a `label` column (1 = real lead); stages never
read it, but the report uses it to show precision/recall and where real leads
were lost at each stage.

## Real models

```bash
ollama pull nomic-embed-text && ollama pull llama3.1:8b
python -m discovery --csv data/synthetic.csv --backend ollama          # local only
ANTHROPIC_API_KEY=... python -m discovery --csv data/synthetic.csv --backend ollama   # + stage 4
```

Ollama's embedding scores differ from the offline stand-in, so re-tune with
`--embed-threshold` (default 0.75 for Ollama, 0.20 offline).

## Reviewer agent (alert / queue / drop)

```bash
python -m discovery.scenarios --out data/review                     # chat with replies + labels
python -m discovery --jsonl data/review_chat.jsonl --db "" --review  # funnel, then review the leads
python -m discovery.evaluate --backend policy                         # score the reviewer against labels
python -m discovery.evaluate --backend ollama --model qwen2.5:7b -v   # with a local model
```

After the funnel, `--review` runs an agent on the top leads (`--review-max`, default 50).
The model reads context through tools (`check_replied`, `get_sender_history`, `get_thread`)
and answers three yes/no questions: answered, urgent, large. `policy.py` turns those answers
plus a repeat-buyer count into the decision; that file is the definition of "valuable to
review". `--review-backend policy` is a fixed procedure with no model; `ollama` uses
`--review-model`. Reviews are saved to the `reviews` table. Notebooks for running Ollama on
Colab are in `colab/`.

## Telegram

```bash
pip install telethon
python -m discovery --telegram @mygroup --api-id 123456 --api-hash abcdef --backend ollama --limit 2000
```

`TelegramSource` in `sources.py` is the only code that touches Telegram. To add
another source (a live listener, a webhook, a Discord export), write any class
with `messages() -> Iterator[Message]`. For live streams, call
`Pipeline.process(Candidate(msg))` per message.

## Layout

| file | role |
|---|---|
| `models.py` | `Message`, `Candidate` shared types |
| `sources.py` | `MessageSource` protocol, CSV / JSONL / Telegram |
| `stages.py` | the four stages (`process(candidate) -> keep?`) |
| `backends.py` | embedders and scorers (offline, Ollama, Claude) |
| `pipeline.py` | runs stages, collects funnel stats and accuracy |
| `factory.py` | the only place that chooses implementations |
| `cost.py`, `storage.py`, `report.py` | cost log, SQLite, terminal report |
| `synthetic.py` | labelled fake chat generator |
| `scenarios.py` | labelled review scenarios: threads, repeat senders, replies |
| `policy.py` | the alert / queue / drop rule (edit here to change the definition) |
| `tools.py`, `agent.py` | reviewer tools and the agent loop |
| `llm.py`, `ollama_llm.py`, `policy_llm.py` | swappable model interface and backends |
| `review_step.py`, `evaluate.py` | `--review` CLI step, and scoring against labels |

## License

MIT, see [LICENSE](LICENSE).
