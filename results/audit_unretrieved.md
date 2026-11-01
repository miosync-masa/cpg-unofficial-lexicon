# Source audit: documents not retrieved by retrieval again or the Wayback Machine

First failure: the last failed request of 2026-10-24. Retrieval again: once more from the same URL on 2026-10-30 and 2026-11-01 (only rows that had failed with HTTP 5xx or a connection error). Wayback Machine: capture closest to 2026-03-13; `HTTP 429` = archive.org refused the capture; dates are MM-DD of 2026.

| item | doc | first failure | again 10-30 | again 11-01 | Wayback |
|---|---|---|---|---|---|
| NAH_004 | 3 | SSLError (10-24) | SSLError | SSLError | 10-25: HTTP 429 ×2, JSONDecodeError; 10-30: HTTP 429 ×2; 11-01: HTTP 429 |
| MET_006 | 2 | HTTP 520 (10-24) | HTTP 520 | HTTP 520 | 10-25: no capture; 10-30: no capture |
| MET_007 | 1 | ConnectTimeout (10-24) | ConnectionError | ConnectionError | 10-25: HTTP 429 ×3; 10-30: HTTP 429 ×2; 11-01: HTTP 429 |
| EEU_011 | 1 | HTTP 500 (10-24) | HTTP 500 | HTTP 500 | 10-25: no capture; 10-30: no capture |
| NET_003 | 2 | ConnectionError (10-24) | ConnectionError | ConnectionError | 10-25: no capture; 10-30: no capture |
| MET_009 | 3 | HTTP 520 (10-24) | HTTP 520 | HTTP 520 | 10-25: no capture; 10-30: HTTP 429 ×2; 11-01: HTTP 429 |
| MET_008 | 2 | ConnectionError (10-24) | ConnectionError | ConnectionError | 10-25: no capture; 10-30: no capture |
| MET_001 | 1 | HTTP 404 (10-24) | — | — | 10-25: no capture; 10-30: no capture |
| SEA_008 | 2 | no request (row recorded) | — | — | 10-25: HTTP 429 ×3; 10-30: HTTP 429 ×2; 11-01: HTTP 429 |
