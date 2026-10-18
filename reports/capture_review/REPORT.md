# Packet capture review: fix future capture, safely recover existing data

Scope: Sessions 03/04 raw cache. Read-only audits, plus one local CPU experiment (E4). No datasets, models or reports were changed.

Scripts and outputs in this folder:

| Script | Output | What it checks |
|---|---|---|
| `packet_audit.py` | `packet_audit.json` | packet types, device vs host gaps, RSSI |
| `format_check.py` | `format_check.json` | pass rates, legacy vs HT packets |
| `gap_tolerance.py` | `gap_tolerance.json` | synthetic single gaps |
| `gap_transplant.py` | `gap_transplant.json` | real gap patterns, and what each gate recovers |

The E4 experiment is `lahacks_v2/e4_gate.py`, with results in `reports/ml_v2/e4_gate.json`.

## 1. Why capture is bad (root cause)

| Finding | Evidence | Confidence |
|---|---|---|
| **The CSI rate depends on frames the router chooses to send to the station.** *(Corrected: the setup is a router AP with the ESP32 station as the receiver, so recording on the station is correct.)* | Every packet comes from the router (`20:50:0D:D0:B8:B5`) and is a 110-byte frame. The station's 100 Hz UDP *uplink* creates no CSI on the station. CSI exists only for the router's *downlink* frames to the station, and the router doesn't send those at a controlled rate. That's why the rate wanders between 15 and 85 Hz. What the 110-byte frames are is unverified: possibly replies or ICMP to the UDP stream, which routers often rate-limit. | Medium-high (mechanism from firmware code; frame identity unverified) |
| Gaps happen on-air, not on serial/USB. | Gaps longer than 250 ms on the device clock ≈ the same gaps on the host clock: 16.4 vs 13.7 per S04 recording. Time spent in gaps: **18.5% in S04**, 6.6% in S03. | High |
| A weak, crowded link makes it worse. | Median RSSI is −64/−66 dBm at 6 ft through the cases. Lower RSSI goes with more gap time (r = −0.36). 89% of frames are HT **40 MHz on 2.4 GHz channel 6** in a crowded venue. | Medium |
| Two packet formats are mixed in one stream. | 89% HT40 and 11% legacy 20 MHz. Same subcarrier shape (r = 0.996), but legacy is **×1.13–1.20 louder**. | High |

## 2. Capture fix (do before recording more)

1. **Give the router a steady stream of frames to send to the station.** Keep the station as the CSI receiver. From a laptop on the same router network, send small UDP packets (or `sudo ping -i 0.01`) to the ESP32's IP at about 100 Hz. The router forwards each one as a downlink frame, and each frame produces one CSI sample. A firmware alternative: have the station ping the router at 100 Hz (echo replies) instead of sending UDP. Also set the collector's `ap_mac` to the router MAC instead of `ANY`. Expected: a steady rate close to the downlink rate you send.
2. Use a less busy channel (scan; usually 1 or 11) and 20 MHz bandwidth if you can, so packet format and scale stay consistent. Keep the boards' line of sight. Try the pokeball cases open vs closed and check RSSI.
3. **Pilot gate before staging falls:** 2 min empty room + 2 min walking. Pass if the median rate is ≥80 Hz and **≥90% of windows pass the frozen v2 gate**. `format_check.py` computes this; point it at the new folder.
4. Optional: the collector already logs `rssi`/`sig_mode`. Have the UI warn live when the rate drops below 30 Hz.

## 3. Maximizing the existing data without bad data

| Question | Answer |
|---|---|
| Are legacy-format packets bad data? | **Not for the v2 model.** Its per-sample gain normalization removes a per-packet scale exactly, so keep them. Dropping them would *lower* pass rates (S04: 28% → 24%). They only hurt the old raw-amplitude models. |
| How long a gap can the v2 model tolerate? | Real gap patterns transplanted onto clean windows flip the final model's decision: **1.3% / 1.0% / 1.7%** for max gaps of 0.25–0.35 / 0.35–0.5 / 0.5–0.75 s. That matches what the current 0.25 s gate already accepts (0.6–1.3%). Above 0.75 s: 10–25% flips. |
| Do gaps become a shortcut? | No. On the relaxed population, interpolation amount or max gap alone predicts the label at AUC 0.41–0.58. |
| What does a gate of **max gap ≤ 0.5 s and edge ≤ 0.25 s** recover? | Normal pass rate S03 **61% → 96%**, S04 **28% → 92%**. Falls with positive windows **13 → 33 of 34**. Normal windows 1,550 → 2,968; positive windows 52 → 194. |
| Does training on those windows help (E4)? | **No.** Same model, labels and threshold rule. AUC: fold A 0.892 → 0.868, fold B 0.893 → 0.890 (v2 population); fold A 0.795 → 0.770, fold B 0.894 → 0.842 (relaxed population). The extra positives are mostly strictly contained S04 windows that include post-fall stillness. |
| Does *evaluating* on them help? | **Yes.** S04 scorable falls go 7 → 29, and normal exposure 3.8 → 9.7 min (S04) and 13.8 → 18.6 min (S03). The current final model catches 24/29 S04 falls on that population, but at ~199 false alerts/h (fold A threshold). |

**Decision**
- **Training:** keep the frozen v2 gate. The relaxed windows aren't bad, but they don't help.
- **Evaluation and live inference:** use the ≤0.5 s gate. The model's output is stable there, and otherwise the detector is blind for 40–70% of the time in conditions like S04.
- **Unchanged:** the frozen v2 config and original pipeline. The relaxed gate exists only as `lahacks_v2.windows.build(max_gap=0.5)`.
- **Next step for the model:** the remaining false-alert rate is the real limit. Fix the capture, then record `session_05` to measure it properly.
