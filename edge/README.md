# blissey_edge: on-device fall detection (QNX / Raspberry Pi)

**Inference runs on TensorFlow Lite** (QNX `python3-tflite-runtime` from oss.qnx.com, as in the project spec). Everything else (CSI parsing, quality gate, features, alerts, uplink) is standard-library Python 3.8+. The training pipeline computes its features with this same code, so device and training can't drift apart.

The logistic model converts exactly to a 5-op float32 TFLite graph (`MAXIMUM → LOG → SUB → FULLY_CONNECTED → LOGISTIC`). Bundles without a `model.tflite`, or `--backend python`, use the stdlib scorer in `model.py`. If `model.json` declares a TFLite model and no runtime is installed, loading fails (`model_status=error`) instead of quietly switching engines.

## Converting a trained model to TFLite (on the Mac)

```sh
pip install flatbuffers ai-edge-litert          # no TensorFlow needed
python3 export_tflite.py ../reports/veri/<JOB>/<run>/deploy ../reports/tflite/<name>
python3 make_pi_kit.py --deploy ../reports/tflite/<name> --wheel <websockets wheel> --out ../reports/pi_kit/<kit>
```

The exporter writes `model.tflite`, adds its SHA-256 to `model.json`, and replays `golden.json` through both scorers. It fails unless every score is within 1e-5 and alerts are identical. Only logistic models convert, because TFLite has no tree ops.

## What's in a deploy bundle (`reports/runs/<run>/deploy/` or the Veri archive)

| File | Purpose |
|---|---|
| `model.json` | Threshold, alert policy, CV summary, provenance, stdlib weights, and the `tflite` entry (file + SHA-256) |
| `model.tflite` | The fall-score model run by `tflite_runtime` on the Pi |
| `golden.json` | Raw packets from a real recording, plus the scores and alerts the training machine produced |
| `blissey_edge/` | Runtime: `csi.py` parse → `window.py` gate/features → `model.py` → `detector.py` alerts |
| `SHA256SUMS` | Checksums of every file in the bundle |

## On the Pi: verify before going live

```sh
cd deploy
python3 -m blissey_edge.selftest model.json golden.json     # must print RESULT: PASS and "backend tflite"
```

The self-test checks the checksums, replays the golden packets through the exact streaming path, and compares every score and alert with the training run. It also prints per-window latency on that device. For TFLite (float32), scores may differ by up to the tolerance in `model.json` (1e-5). Alerts must still match exactly.

## Running on the Pi (with the verified QNX transport)

The Pi runs **two processes**:
- **this service** reads the AP receiver's serial port, detects falls, and enqueues one event per episode plus a health snapshot
- **Emilia's transport** (`blissey-beacon` branch `codex/qnx-websocket-client`, `edge/qnx`, installed at `/data/home/qnxuser/blissey-beacon/transport`) sends them over WSS with the Pi's registered token

**Wiring:** the Pi's USB goes to the **AP receiver** ESP32 (Wi-Fi MAC `20:50:0D:D0:B8:B5`, firmware `blissey-csi-1` active_ap), which prints CSI. The **STA transmitter** (`20:50:0D:CC:80:80`) needs power only; never open its serial port.

```sh
cd blissey_pi_v1
python3 -m blissey_edge.pi_check                           # Python, TLS, clock, model selftest, lists serial devices
python3 -m blissey_edge.pi_check --serial /dev/serusbN     # 10 s live: expect ~80-100 Hz, roles {'AP'}, one source MAC
python3 -m blissey_edge.service --serial /dev/serusbN --no-network                  # events printed only
# live: terminal 1 = transport (Emilia's), terminal 2 = this producer
sh /data/home/qnxuser/blissey-beacon/transport/run-client.sh /data/home/qnxuser/blissey-beacon/config/device.json
python3 -m blissey_edge.service --serial /dev/serusbN --qnx-config /data/home/qnxuser/blissey-beacon/config/device.json
```

In `--qnx-config` mode, `device_id` / `room_id`, the SQLite outbox and the health-file paths all come from the transport's `device.json`. No token is read by this service. Add `--clock-synced` only once the Pi's clock service is verified.

Standalone mode, without the QNX transport, also exists. This service then sends itself over its own stdlib WebSocket (`--token-file`, `--make-token`), but don't run both senders for the same device.

| Module | Role |
|---|---|
| `csi.py`, `serial_source.py` | Parse ESP32 lines; read the USB serial device (termios, or set the baud with `stty`) |
| `window.py`, `model.py`, `detector.py` | Quality gate → pv1 features → model (TFLite or stdlib) → 2-of-3 vote alerts |
| `episodes.py` | One event per fall episode; suppresses alerts for 60 s after each event. Builds the exact 7-field v2 event. |
| `contracts.py` | Local mirror of the backend schemas; nothing invalid is ever queued |
| `outbox.py` | Durable, fsynced outbox; removal only on `event_ack`; dead letters for non-retryable rejections |
| `ws.py`, `uplink.py` | Stdlib WebSocket client (TLS-verified); Bearer auth; health every 10 s; reconnect with backoff |
| `service.py` | Ties it together; `--replay file --no-network` for testing without hardware |
| `selftest.py`, `pi_check.py` | Golden-output check; Phase 1 platform check |

**Credential safety:** the token stays in a 0600 file on the Pi and is never logged or sent in a URL or payload. A rejected token stops sending and retries every 10 min, while detection and the outbox keep working.

Tests (laptop): `cd edge && python3 -m unittest test_edge`. They run a fake backend mirroring `apps/backend/src/ingestion/routes.ts`, and check payloads against the real `blissey-beacon` schemas and fixtures.
