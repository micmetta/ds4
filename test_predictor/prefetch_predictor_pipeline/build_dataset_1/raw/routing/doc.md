# Trace di routing per prompt

Questa cartella contiene 100 file `prompt_XXXX.routing.jsonl`, uno per prompt
del corpus V0. Ogni riga descrive un evento di routing DS4 durante prefill o
decode.

Per BUILD_1 il preprocessing usa soltanto gli eventi `decode` routed e non-hash.
Da essi ricava feature 1–3: routing Top-6 corrente, raw router logits Top-16 e
`token_id`, oltre al target di routing del token successivo.
