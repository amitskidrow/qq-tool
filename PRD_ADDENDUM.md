# PRD Addendum — v1.2 Updates

This addendum amends `qq/prd.md` with the latest decisions.

Key Changes

- Install: qq remains Docker-free; Qdrant is assumed to run locally as a container/service on `http://localhost:6333` with no auth.
- Token caps: per-account daily caps (not per-session), reset at midnight IST (Asia/Kolkata):
  - gpt-5: 250k tokens/day
  - gpt-5-mini/gpt-5-nano: 2M tokens/day
  - Autoswitch to `gemini-2.5-pro` upon cap exceedance; record the switch.
- Models: no mapping; supported names are exactly `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gemini-2.5-pro`.
- OPA: ship integration as optional; enabled by default if available.

Config Additions

```yaml
vector_store:
  url: http://localhost:6333
  collection: qq_data
usage:
  mode: daily
  timezone: Asia/Kolkata
```

New/Clarified Error Codes

- `ERR_VECTOR_UNAVAILABLE`: Qdrant not reachable at configured endpoint.
- `ERR_TOKEN_CAP_EXCEEDED`: daily cap reached; autoswitch performed or request denied.

