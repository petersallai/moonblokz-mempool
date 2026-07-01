# moonblokz-mempool

Bounded, `no_std`, no-alloc MoonBlokz mempool — a compacted byte-buffer + fixed-size index that holds pending transactions within a fixed memory budget (FR30).

- `no_std`, no-alloc, embassy-free.
- Leaf crate: depends only on `moonblokz-chain-types` (for `TransactionView<'_>`) and `rand_xoshiro` (Story 2.2 eviction PRNG). **No** dependency on `moonblokz-blockchain`.
- `Mempool<COMPACT_BYTES, MAX_ENTRIES>` — defaults per architecture §5: `COMPACT_BYTES = 20160`, `MAX_ENTRIES = 128`. Node-roster capacity is owned by blockchain/vote, not the mempool storage contract.
- Index entries store `hash_crc32` (IEEE CRC32 over the canonical transaction hash) as a no-alloc lookup/replenishment fingerprint, not CRC32 of serialized transaction bytes.
- Index entries also store `transaction_fee`, supplied at admission after blockchain validation has resolved any UTXO-backed complex fee.
- `top_n_for_exchange(n)` yields `(TransactionView<'_>, transaction_fee)` pairs in FR45 priority order without allocation, reusing the stored fee instead of recalculating it.

Implementation tracked story-by-story in `_bmad-output/implementation-artifacts/sprint-status.yaml`.
