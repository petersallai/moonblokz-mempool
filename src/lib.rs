#![no_std]

//! # moonblokz-mempool
//!
//! Bounded `no_std` MoonBlokz mempool: a compacted byte-buffer + fixed-size
//! index that holds pending transactions within a fixed memory budget (FR30).
//!
//! **Leaf-crate discipline.** Depends only on `moonblokz-chain-types`
//! (`TransactionView<'_>`) and `rand_xoshiro` (Story 2.2 eviction PRNG).
//! **No** dependency on `moonblokz-blockchain` — the blockchain hands a
//! caller-derived `sub_seed: u64` to [`Mempool::new`] and the mempool runs
//! standalone.
//!
//! ## Storage contract (FR30)
//!
//! - `compact_buffer: [u8; COMPACT_BYTES]` — contiguous serialized transaction
//!   bytes, gap-free from index 0 to `byte_usage`.
//! - `index: [Option<IndexEntry>; MAX_ENTRIES]` — fixed-capacity index with
//!   `(start, length, hash_crc32, transaction_fee, expiry_sequence,
//!   is_deferred, is_own)` per entry. The contiguous-storage invariant is
//!   enforced after every mutation.
//!
//! ## Story scope
//!
//! Story 2.1 implements the storage contract (admission when space is
//! available, lookup, removal-with-compaction, eligibility flag toggling,
//! borrowed iteration). Story 2.2 layers the FR33 ownership-differentiated
//! capacity-pressure eviction on top.

use moonblokz_chain_types::{BlockView, TransactionView, calculate_hash};
use rand_xoshiro::Xoshiro256PlusPlus;
use rand_xoshiro::rand_core::SeedableRng;

/// Sentinel for transactions with no byte-local sequence dependency.
///
/// FR53 reserves `u32::MAX` as an invalid block sequence in MVP, so it is a
/// safe in-memory marker for "do not expire by anchor/window sequence".
pub const NO_EXPIRY_SEQUENCE: u32 = u32::MAX;

/// Index entry pointing at a transaction inside `compact_buffer`.
///
/// This is deliberately private storage metadata rather than public API; code
/// in this module uses direct field access to avoid embedded-unfriendly accessor
/// boilerplate. `Copy + Clone` are derived solely so `[None; MAX_ENTRIES]`
/// array initialization works in `const fn`-style construction. The layout
/// stays deliberately straightforward for Story 2.1; Story 2.2 may compact this
/// further if the architecture §6.8 per-entry budget becomes load-bearing.
#[derive(Copy, Clone)]
struct IndexEntry {
    start: u16,
    length: u16,
    // Hash lookup uses `hash_crc32` as a prefilter. Priority ordering and
    // later block assembly consume `transaction_fee`; it is supplied by the
    // blockchain caller at admission time so UTXO-backed complex fees are not
    // recomputed during iteration.
    hash_crc32: u32,
    transaction_fee: u64,
    #[allow(dead_code)]
    expiry_sequence: u32,
    is_deferred: bool,
    #[allow(dead_code)]
    is_own: bool,
}

/// Outcome of [`Mempool::try_add`].
///
/// Story 2.1 returns only `Admitted` on the happy path or `Rejected` when
/// the buffer / index is full. Story 2.2 introduces the FR33 capacity-
/// pressure eviction paths that can also return `Rejected` (own-only Case 1b
/// fallback) or `Admitted` after eviction (Cases 1a / 1b / 2).
pub enum AddResult {
    /// The transaction was accepted into the mempool.
    Admitted,
    /// The transaction was not accepted. Story 2.1: insufficient buffer or
    /// index capacity. Story 2.2: also the FR33 Case 1b "own-only mempool"
    /// fallback where the arriving transaction is the drawn eviction target.
    Rejected,
}

/// Bounded MoonBlokz mempool.
///
/// Const generics:
/// - `COMPACT_BYTES`: byte-buffer capacity (architecture §5 default: 20160).
/// - `MAX_ENTRIES`: maximum number of indexed transactions (default: 128).
#[allow(dead_code)] // prng consumed by Story 2.2 capacity-pressure logic
pub struct Mempool<const COMPACT_BYTES: usize, const MAX_ENTRIES: usize> {
    compact_buffer: [u8; COMPACT_BYTES],
    index: [Option<IndexEntry>; MAX_ENTRIES],
    prng: Xoshiro256PlusPlus,
    byte_usage: u16,
    entry_count: u8,
    own_node_id: u32,
}

impl<const COMPACT_BYTES: usize, const MAX_ENTRIES: usize> Mempool<COMPACT_BYTES, MAX_ENTRIES> {
    /// Constructs an empty `Mempool` seeded for deterministic-replay
    /// (Story 2.2 eviction PRNG).
    ///
    /// Per FR59, nothing is recovered from durable storage — the mempool is
    /// empty on restart by design.
    pub fn new(sub_seed: u64, own_node_id: u32) -> Self {
        assert!(
            COMPACT_BYTES <= u16::MAX as usize,
            "COMPACT_BYTES must fit in u16 offsets"
        );
        assert!(
            MAX_ENTRIES <= u8::MAX as usize,
            "MAX_ENTRIES must fit in the u8 entry_count"
        );

        Self {
            compact_buffer: [0u8; COMPACT_BYTES],
            index: [None; MAX_ENTRIES],
            prng: Xoshiro256PlusPlus::seed_from_u64(sub_seed),
            byte_usage: 0,
            entry_count: 0,
            own_node_id,
        }
    }

    /// Returns the current number of indexed transactions.
    pub fn entry_count(&self) -> u8 {
        self.entry_count
    }

    /// Returns the current byte-buffer usage. Useful for the FR30 contiguous-
    /// storage invariant verification.
    pub fn byte_usage(&self) -> u16 {
        self.byte_usage
    }

    /// Attempts to admit `tx` into the mempool.
    ///
    /// Story 2.1: succeeds when both (a) `byte_usage + tx.len() ≤ COMPACT_BYTES`
    /// and (b) `entry_count < MAX_ENTRIES`. Otherwise returns `Rejected`.
    /// Story 2.2 replaces the `Rejected` paths with FR33 capacity-pressure
    /// eviction.
    ///
    /// The buffer-append + index-write pair is structurally atomic: both
    /// updates happen after the bounds check, with no intervening panic
    /// point.
    pub fn try_add(
        &mut self,
        tx: TransactionView<'_>,
        transaction_fee: u64,
        is_deferred: bool,
    ) -> AddResult {
        let bytes = tx.as_bytes();
        let tx_len = bytes.len();

        if self.byte_usage as usize + tx_len > COMPACT_BYTES {
            return AddResult::Rejected;
        }
        if (self.entry_count as usize) >= MAX_ENTRIES {
            return AddResult::Rejected;
        }

        let slot = match find_free_slot(&self.index) {
            Some(s) => s,
            None => return AddResult::Rejected, // unreachable given entry_count check
        };
        let start = self.byte_usage as usize;
        self.compact_buffer[start..start + tx_len].copy_from_slice(bytes);
        let tx_hash = calculate_hash(bytes);

        self.index[slot] = Some(IndexEntry {
            start: start as u16,
            length: tx_len as u16,
            hash_crc32: crc32_ieee(&tx_hash),
            transaction_fee,
            expiry_sequence: transaction_expiry_sequence(&tx),
            is_deferred,
            is_own: is_own_transaction(&tx, self.own_node_id),
        });

        self.byte_usage += tx_len as u16;
        self.entry_count += 1;
        AddResult::Admitted
    }

    /// Lookup by canonical transaction hash. Returns a borrowed view into
    /// `compact_buffer` without copying.
    pub fn get_by_hash(&self, hash: &[u8; 32]) -> Option<TransactionView<'_>> {
        match self.find_by_hash(hash) {
            Some(tx_bytes) => TransactionView::from_bytes(tx_bytes),
            None => None,
        }
    }

    /// `true` iff a transaction with the given canonical hash is present.
    pub fn contains(&self, hash: &[u8; 32]) -> bool {
        self.find_by_hash(hash).is_some()
    }

    /// Returns the serialized bytes for a transaction already stored in the
    /// compact buffer. Private helper so `get_by_hash` and `contains` share
    /// exactly one scan implementation.
    ///
    /// `hash_crc32` is an index prefilter only. CRC32 collisions are possible,
    /// so a matching CRC is always followed by the full canonical hash check.
    fn find_by_hash(&self, hash: &[u8; 32]) -> Option<&[u8]> {
        let hash_crc32 = crc32_ieee(hash);
        for entry in self.index.iter().filter_map(|entry| entry.as_ref()) {
            if entry.hash_crc32 != hash_crc32 {
                continue;
            }

            let start = entry.start as usize;
            let len = entry.length as usize;
            let tx_bytes = &self.compact_buffer[start..start + len];
            if &calculate_hash(tx_bytes) == hash {
                return Some(tx_bytes);
            }
        }
        None
    }

    /// Removes mempool entries whose serialized transaction bytes appear in an
    /// accepted transaction block, then compacts the buffer to preserve the FR30
    /// contiguous-storage invariant. *(Blockchain-driven invocation: Epic 7 /
    /// FR32.)*
    ///
    /// Accepts a `BlockView` directly so callers do not need to materialize a
    /// temporary array of transaction hashes on stack/RAM-constrained targets.
    /// Non-transaction blocks are a no-op.
    pub fn confirm_by_block_acceptance(&mut self, accepted: &BlockView<'_>) {
        let accepted_txs = match accepted.transactions() {
            Some(transactions) => transactions,
            None => return,
        };
        let mut removed = false;

        // Pass 1: null out matching slots; the byte-buffer compaction happens
        // in pass 2. Compare canonical serialized transaction bytes directly:
        // the accepted block already carries the exact transaction bytes.
        for entry_opt in self.index.iter_mut() {
            if let Some(entry) = entry_opt {
                let start = entry.start as usize;
                let len = entry.length as usize;
                let tx_bytes = &self.compact_buffer[start..start + len];
                if accepted_txs
                    .iter()
                    .any(|accepted_tx| accepted_tx.as_bytes() == tx_bytes)
                {
                    *entry_opt = None;
                    removed = true;
                }
            }
        }

        if removed {
            self.compact_after_removal();
        }
    }

    /// Recomputes eligibility flags via `balance_check(initializer) -> balance`.
    /// *(Blockchain-driven invocation: Epic 7 / FR15.)*
    ///
    /// Story 2.1 implements the FR15 hand-off for node-transfer + registration
    /// kinds (balance ≥ amount + fee, balance ≥ registration_price + fee).
    /// `ComplexTransaction` eligibility is the multi-input UTXO check from
    /// ADR-016 and lands in Story 7.3 / 7.5; Story 2.1 treats complex
    /// transactions as **always eligible** here as a temporary stub.
    pub fn recheck_eligibility<F>(&mut self, mut balance_check: F)
    where
        F: FnMut(u32) -> u64,
    {
        for entry_opt in self.index.iter_mut() {
            if let Some(entry) = entry_opt {
                let start = entry.start as usize;
                let len = entry.length as usize;
                let tx_bytes = &self.compact_buffer[start..start + len];
                let tx = match TransactionView::from_bytes(tx_bytes) {
                    Some(v) => v,
                    None => continue,
                };

                let eligible = if let Some(nt) = tx.as_node_transfer() {
                    balance_check(nt.initializer())
                        >= nt.amount().saturating_add(entry.transaction_fee)
                } else if let Some(reg) = tx.as_registration() {
                    balance_check(reg.initializer())
                        >= reg
                            .registration_price()
                            .saturating_add(entry.transaction_fee)
                } else {
                    // Complex transactions: Story 2.1 stub treats as eligible.
                    // Proper multi-input UTXO eligibility lands in Story 7.3 / 7.5.
                    true
                };

                entry.is_deferred = !eligible;
            }
        }
    }

    /// Yields borrowed `TransactionView<'_>`s for all currently-eligible
    /// (non-deferred) mempool entries. No allocation, no state mutation.
    pub fn eligible_iter(&self) -> EligibleIter<'_, COMPACT_BYTES, MAX_ENTRIES> {
        EligibleIter {
            mempool: self,
            idx: 0,
        }
    }

    /// Yields up to `n` borrowed transactions plus their already-resolved
    /// transaction fees in the deterministic FR45 / FR43 mempool priority
    /// order: fee-per-byte descending, own before other only among equal
    /// fee-per-byte candidates, ascending `hash_crc32`, then lexicographic
    /// transaction bytes.
    ///
    /// The fee is supplied at admission time and stored in `IndexEntry` so
    /// UTXO-backed complex fees do not need to be recomputed while selecting
    /// block/exchange candidates.
    pub fn top_n_for_exchange(&self, n: usize) -> TopNIter<'_, COMPACT_BYTES, MAX_ENTRIES> {
        TopNIter {
            mempool: self,
            remaining: n,
            yielded_bits: [0; 4],
        }
    }

    /// Internal: compacts `compact_buffer` and `index` so occupied entries
    /// form a contiguous prefix from offset 0 to `byte_usage`. Called after
    /// every `confirm_by_block_acceptance` mutation.
    fn compact_after_removal(&mut self) {
        // Compact in place to avoid a second `[u8; COMPACT_BYTES]` stack
        // buffer on embedded targets. Survivors are processed in ascending
        // original `start` order by repeatedly selecting the next unprocessed
        // entry; updated entries get smaller/equal starts and are excluded by
        // `last_original_start`.
        let survivor_total = self.index.iter().filter(|entry| entry.is_some()).count();
        let old_byte_usage = self.byte_usage as usize;
        let mut write_offset = 0u16;
        let mut last_original_start: Option<u16> = None;

        for _ in 0..survivor_total {
            let mut next_slot: Option<usize> = None;
            let mut next_start = u16::MAX;

            for (slot, entry_opt) in self.index.iter().enumerate() {
                let entry = match entry_opt {
                    Some(entry) => entry,
                    None => continue,
                };
                if last_original_start.is_some_and(|last| entry.start <= last) {
                    continue;
                }
                if entry.start < next_start {
                    next_start = entry.start;
                    next_slot = Some(slot);
                }
            }

            let slot = next_slot.expect("survivor count and index entries must agree");
            let mut entry = self.index[slot].expect("selected slot must contain an entry");
            let original_start = entry.start;
            let src_start = original_start as usize;
            let src_len = entry.length as usize;
            let dst_start = write_offset as usize;

            if src_start != dst_start {
                self.compact_buffer
                    .copy_within(src_start..src_start + src_len, dst_start);
            }

            entry.start = write_offset;
            self.index[slot] = Some(entry);
            write_offset += entry.length;
            last_original_start = Some(original_start);
        }

        let new_byte_usage = write_offset as usize;
        if new_byte_usage < old_byte_usage {
            self.compact_buffer[new_byte_usage..old_byte_usage].fill(0);
        }

        self.byte_usage = write_offset;
        self.entry_count = survivor_total as u8;
    }
}

/// Iterator over eligible (non-deferred) mempool entries.
pub struct EligibleIter<'a, const COMPACT_BYTES: usize, const MAX_ENTRIES: usize> {
    mempool: &'a Mempool<COMPACT_BYTES, MAX_ENTRIES>,
    idx: usize,
}

impl<'a, const COMPACT_BYTES: usize, const MAX_ENTRIES: usize> Iterator
    for EligibleIter<'a, COMPACT_BYTES, MAX_ENTRIES>
{
    type Item = TransactionView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        while self.idx < MAX_ENTRIES {
            let slot = self.idx;
            self.idx += 1;
            if let Some(entry) = &self.mempool.index[slot] {
                if entry.is_deferred {
                    continue;
                }
                let start = entry.start as usize;
                let len = entry.length as usize;
                let tx_bytes = &self.mempool.compact_buffer[start..start + len];
                return TransactionView::from_bytes(tx_bytes);
            }
        }
        None
    }
}

/// Iterator over the top eligible mempool entries in FR45 priority order.
///
/// The iterator keeps only a 256-bit yielded-slot bitmap (32 bytes) and scans
/// the fixed index on each `next()` call. This avoids allocation and avoids a
/// temporary sorted array on RAM-constrained targets.
pub struct TopNIter<'a, const COMPACT_BYTES: usize, const MAX_ENTRIES: usize> {
    mempool: &'a Mempool<COMPACT_BYTES, MAX_ENTRIES>,
    remaining: usize,
    yielded_bits: [u64; 4],
}

impl<'a, const COMPACT_BYTES: usize, const MAX_ENTRIES: usize> Iterator
    for TopNIter<'a, COMPACT_BYTES, MAX_ENTRIES>
{
    type Item = (TransactionView<'a>, u64);

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let mut best_slot: Option<usize> = None;

        for slot in 0..MAX_ENTRIES {
            if self.is_yielded(slot) {
                continue;
            }

            let Some(entry) = self.mempool.index[slot].as_ref() else {
                continue;
            };
            if entry.is_deferred {
                continue;
            }

            let tx_bytes = entry_bytes(self.mempool, entry);
            if TransactionView::from_bytes(tx_bytes).is_none() {
                continue;
            }

            let is_better = match best_slot {
                Some(current_best_slot) => {
                    let best_entry = self.mempool.index[current_best_slot]
                        .as_ref()
                        .expect("selected best slot must stay occupied");
                    let best_bytes = entry_bytes(self.mempool, best_entry);
                    priority_precedes(
                        entry,
                        tx_bytes,
                        slot,
                        best_entry,
                        best_bytes,
                        current_best_slot,
                    )
                }
                None => true,
            };

            if is_better {
                best_slot = Some(slot);
            }
        }

        let slot = best_slot?;
        self.mark_yielded(slot);
        self.remaining -= 1;

        let entry = self.mempool.index[slot]
            .as_ref()
            .expect("selected top slot must stay occupied");
        TransactionView::from_bytes(entry_bytes(self.mempool, entry))
            .map(|tx| (tx, entry.transaction_fee))
    }
}

impl<'a, const COMPACT_BYTES: usize, const MAX_ENTRIES: usize>
    TopNIter<'a, COMPACT_BYTES, MAX_ENTRIES>
{
    fn is_yielded(&self, slot: usize) -> bool {
        (self.yielded_bits[slot / 64] & (1u64 << (slot % 64))) != 0
    }

    fn mark_yielded(&mut self, slot: usize) {
        self.yielded_bits[slot / 64] |= 1u64 << (slot % 64);
    }
}

// =================== internal helpers ===================

fn find_free_slot<const N: usize>(index: &[Option<IndexEntry>; N]) -> Option<usize> {
    for (i, e) in index.iter().enumerate() {
        if e.is_none() {
            return Some(i);
        }
    }
    None
}

fn entry_bytes<'a, const COMPACT_BYTES: usize, const MAX_ENTRIES: usize>(
    mempool: &'a Mempool<COMPACT_BYTES, MAX_ENTRIES>,
    entry: &IndexEntry,
) -> &'a [u8] {
    let start = entry.start as usize;
    let len = entry.length as usize;
    &mempool.compact_buffer[start..start + len]
}

fn priority_precedes(
    candidate_entry: &IndexEntry,
    candidate_bytes: &[u8],
    candidate_slot: usize,
    best_entry: &IndexEntry,
    best_bytes: &[u8],
    best_slot: usize,
) -> bool {
    let candidate_scaled = (candidate_entry.transaction_fee as u128) * (best_entry.length as u128);
    let best_scaled = (best_entry.transaction_fee as u128) * (candidate_entry.length as u128);
    if candidate_scaled != best_scaled {
        return candidate_scaled > best_scaled;
    }

    if candidate_entry.is_own != best_entry.is_own {
        return candidate_entry.is_own;
    }

    if candidate_entry.hash_crc32 != best_entry.hash_crc32 {
        return candidate_entry.hash_crc32 < best_entry.hash_crc32;
    }

    match candidate_bytes.cmp(best_bytes) {
        core::cmp::Ordering::Less => true,
        core::cmp::Ordering::Greater => false,
        core::cmp::Ordering::Equal => candidate_slot < best_slot,
    }
}

/// Computes the byte-local sequence dependency used for future window-expiry
/// removal.
///
/// Node-transfer transactions carry a direct `anchor_sequence`. Registration
/// transactions carry no anchor and therefore do not expire by this marker.
/// Complex transactions may have multiple balance inputs, so the minimum
/// balance-input `anchor_sequence` is enough to decide whether any such input
/// has fallen before the active-chain tail. UTXO inputs reference prior
/// transactions by hash/output index; their containing block sequence is known
/// only to the blockchain / UTXO cache, not to this byte-local mempool index.
/// UTXO-only and zero-input complex transactions therefore use
/// [`NO_EXPIRY_SEQUENCE`] here and require later blockchain-context-driven
/// removal if needed.
fn transaction_expiry_sequence(tx: &TransactionView<'_>) -> u32 {
    if let Some(nt) = tx.as_node_transfer() {
        return nt.anchor_sequence();
    }
    if tx.as_registration().is_some() {
        return NO_EXPIRY_SEQUENCE;
    }
    let Some(complex) = tx.as_complex() else {
        return NO_EXPIRY_SEQUENCE;
    };

    let mut expiry_sequence = NO_EXPIRY_SEQUENCE;
    for input in complex.inputs() {
        if let Some(balance) = input.as_balance() {
            let anchor = balance.anchor_sequence();
            if anchor < expiry_sequence {
                expiry_sequence = anchor;
            }
        }
    }
    expiry_sequence
}

/// Classifies whether `tx` is own from the local node's perspective.
///
/// Policy resolved during Story 2.1 code review:
/// - node-transfer / registration: own when `initializer == own_node_id`;
/// - complex: own when it has at least one input and any balance input
///   initializer or balance output receiver equals `own_node_id`;
/// - UTXO-only / zero-input complex transactions are non-own.
fn is_own_transaction(tx: &TransactionView<'_>, own_node_id: u32) -> bool {
    if let Some(nt) = tx.as_node_transfer() {
        return nt.initializer() == own_node_id;
    }
    if let Some(reg) = tx.as_registration() {
        return reg.initializer() == own_node_id;
    }
    let Some(complex) = tx.as_complex() else {
        return false;
    };

    if complex.input_count() == 0 {
        return false;
    }

    for input in complex.inputs() {
        if input
            .as_balance()
            .is_some_and(|balance| balance.initializer() == own_node_id)
        {
            return true;
        }
    }
    for output in complex.outputs() {
        if output
            .as_balance()
            .is_some_and(|balance| balance.receiver() == own_node_id)
        {
            return true;
        }
    }
    false
}

/// CRC32 over IEEE 802.3 polynomial (`0xEDB88320`) — same default as
/// `crc32fast`. Mempool index entries store this over the canonical
/// transaction hash, not over the full transaction bytes.
fn crc32_ieee(bytes: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB8_8320;
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if (crc & 1) != 0 {
                (crc >> 1) ^ POLY
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

// =================== tests ===================

#[cfg(test)]
mod tests {
    use super::*;
    use moonblokz_chain_types::{
        ComplexTransaction, HEADER_SIZE, NODE_TRANSFER_SIZE, NodeTransfer,
        PAYLOAD_TYPE_TRANSACTION, REGISTRATION_SIZE, Registration,
    };

    // Test instance parameters: small COMPACT_BYTES and MAX_ENTRIES so we
    // can exercise the buffer-full / index-full rejection paths cheaply.
    const TEST_COMPACT_BYTES: usize = 1024;
    const TEST_MAX_ENTRIES: usize = 8;
    type TestMempool = Mempool<TEST_COMPACT_BYTES, TEST_MAX_ENTRIES>;

    fn sample_node_transfer(vote: u32, initializer: u32, amount: u64) -> NodeTransfer {
        sample_node_transfer_with_fee(vote, initializer, amount, 1)
    }

    fn sample_node_transfer_with_fee(
        vote: u32,
        initializer: u32,
        amount: u64,
        fee: u32,
    ) -> NodeTransfer {
        sample_node_transfer_with_anchor_and_fee(vote, 0, initializer, amount, fee)
    }

    fn sample_node_transfer_with_anchor(
        vote: u32,
        anchor_sequence: u32,
        initializer: u32,
        amount: u64,
    ) -> NodeTransfer {
        sample_node_transfer_with_anchor_and_fee(vote, anchor_sequence, initializer, amount, 1)
    }

    fn sample_node_transfer_with_anchor_and_fee(
        vote: u32,
        anchor_sequence: u32,
        initializer: u32,
        amount: u64,
        fee: u32,
    ) -> NodeTransfer {
        let sig = [0xAA; 64];
        NodeTransfer::new(vote, anchor_sequence, initializer, 0, amount, fee, 0, &sig)
    }

    fn sample_registration(new_node_id: u32) -> Registration {
        sample_registration_with_fee(new_node_id, 1)
    }

    fn sample_registration_with_fee(new_node_id: u32, fee: u64) -> Registration {
        let pub_key = [0xBB; 32];
        let sig = [0xCC; 64];
        Registration::new(0, 0, new_node_id, 100, fee, &pub_key, &sig, &sig)
    }

    fn test_fee(tx: &TransactionView<'_>) -> u64 {
        if let Some(nt) = tx.as_node_transfer() {
            nt.fee() as u64
        } else if let Some(reg) = tx.as_registration() {
            reg.fee()
        } else if let Some(complex) = tx.as_complex() {
            let mut inputs = 0u64;
            for input in complex.inputs() {
                if let Some(balance) = input.as_balance() {
                    inputs = inputs.saturating_add(balance.amount());
                }
            }
            let mut outputs = 0u64;
            for output in complex.outputs() {
                if let Some(balance) = output.as_balance() {
                    outputs = outputs.saturating_add(balance.amount());
                } else if let Some(utxo) = output.as_utxo() {
                    outputs = outputs.saturating_add(utxo.amount());
                }
            }
            inputs.saturating_sub(outputs)
        } else {
            0
        }
    }

    fn try_add_test_tx<const C: usize, const E: usize>(
        mp: &mut Mempool<C, E>,
        tx: TransactionView<'_>,
        is_deferred: bool,
    ) -> AddResult {
        let transaction_fee = test_fee(&tx);
        mp.try_add(tx, transaction_fee, is_deferred)
    }

    fn set_entry_hash_crc32<const C: usize, const E: usize>(
        mp: &mut Mempool<C, E>,
        tx_bytes: &[u8],
        hash_crc32: u32,
    ) {
        for slot in 0..E {
            let matches = match mp.index[slot] {
                Some(entry) => entry_bytes(mp, &entry) == tx_bytes,
                None => false,
            };
            if matches {
                mp.index[slot]
                    .as_mut()
                    .expect("matched slot must remain occupied")
                    .hash_crc32 = hash_crc32;
                return;
            }
        }
        panic!("test transaction must be present in mempool");
    }

    fn transaction_block_view<'a>(buffer: &'a mut [u8], txs: &[&[u8]]) -> BlockView<'a> {
        const PAYLOAD_TYPE_OFFSET: usize = 13;

        assert!(txs.len() <= u16::MAX as usize);
        assert!(buffer.len() >= HEADER_SIZE + 2);

        buffer.fill(0);
        buffer[0] = 1; // version
        buffer[PAYLOAD_TYPE_OFFSET] = PAYLOAD_TYPE_TRANSACTION;

        let mut offset = HEADER_SIZE;
        buffer[offset..offset + 2].copy_from_slice(&(txs.len() as u16).to_le_bytes());
        offset += 2;

        for tx in txs {
            let end = offset + tx.len();
            assert!(end <= buffer.len());
            buffer[offset..end].copy_from_slice(tx);
            offset = end;
        }

        match BlockView::from_bytes(&buffer[..offset]) {
            Ok(view) => view,
            Err(_) => panic!("test transaction block must be valid"),
        }
    }

    #[test]
    #[should_panic(expected = "COMPACT_BYTES must fit in u16 offsets")]
    fn new_rejects_unsupported_compact_bytes() {
        let _mp: Mempool<65536, 8> = Mempool::new(0x1234, 42);
    }

    #[test]
    #[should_panic(expected = "MAX_ENTRIES must fit in the u8 entry_count")]
    fn new_rejects_unsupported_entry_count() {
        let _mp: Mempool<1024, 256> = Mempool::new(0x1234, 42);
    }

    #[test]
    fn new_empty_state() {
        let mp = TestMempool::new(0x1234, 42);
        assert_eq!(mp.entry_count(), 0);
        assert_eq!(mp.byte_usage(), 0);
        assert!(mp.index.iter().all(|e| e.is_none()));
    }

    #[test]
    fn try_add_admits_when_space_available() {
        let mut mp = TestMempool::new(0x1234, 42);
        let nt = sample_node_transfer(0, 1, 100);
        let tx = TransactionView::from_bytes(nt.as_bytes()).unwrap();

        match try_add_test_tx(&mut mp, tx, false) {
            AddResult::Admitted => {}
            AddResult::Rejected => panic!("first add must succeed on empty mempool"),
        }
        assert_eq!(mp.entry_count(), 1);
        assert!(mp.byte_usage() > 0);
    }

    #[test]
    fn try_add_rejects_on_index_full() {
        let mut mp = TestMempool::new(0x1234, 42);
        let nt = sample_node_transfer(0, 1, 100);
        let tx_bytes = nt.as_bytes();

        // Fill all MAX_ENTRIES slots.
        for _ in 0..TEST_MAX_ENTRIES {
            let tx = TransactionView::from_bytes(tx_bytes).unwrap();
            assert!(matches!(
                try_add_test_tx(&mut mp, tx, false),
                AddResult::Admitted
            ));
        }
        // Next add must be Rejected (index full; buffer might also be full).
        let tx = TransactionView::from_bytes(tx_bytes).unwrap();
        assert!(matches!(
            try_add_test_tx(&mut mp, tx, false),
            AddResult::Rejected
        ));
        assert_eq!(mp.entry_count() as usize, TEST_MAX_ENTRIES);
    }

    #[test]
    fn try_add_rejects_on_buffer_overflow() {
        // Use a single-entry-fitting capacity to force buffer-overflow path
        // before index-full path.
        const TINY: usize = 200; // smaller than 2 × NODE_TRANSFER_SIZE (~202)
        let mut mp: Mempool<TINY, 8> = Mempool::new(0x1234, 42);
        let nt = sample_node_transfer(0, 1, 100);
        let tx = TransactionView::from_bytes(nt.as_bytes()).unwrap();

        // First add succeeds.
        assert!(matches!(
            try_add_test_tx(&mut mp, tx, false),
            AddResult::Admitted
        ));
        // Second add fails on buffer overflow (NODE_TRANSFER_SIZE = 101; 101+101 > 200).
        let tx2 = TransactionView::from_bytes(nt.as_bytes()).unwrap();
        assert!(matches!(
            try_add_test_tx(&mut mp, tx2, false),
            AddResult::Rejected
        ));
        assert_eq!(mp.entry_count(), 1);
    }

    #[test]
    fn get_by_hash_returns_borrowed_view() {
        let mut mp = TestMempool::new(0x1234, 42);
        let nt = sample_node_transfer(0, 7, 1234);
        let tx_bytes = nt.as_bytes();
        let tx = TransactionView::from_bytes(tx_bytes).unwrap();
        let hash = calculate_hash(tx_bytes);

        assert!(matches!(
            try_add_test_tx(&mut mp, tx, false),
            AddResult::Admitted
        ));

        let view = mp
            .get_by_hash(&hash)
            .expect("just-added tx must be findable");
        // The returned view must borrow from `compact_buffer` directly —
        // its bytes pointer must be inside the mempool's buffer range.
        let buffer_start = mp.compact_buffer.as_ptr() as usize;
        let buffer_end = buffer_start + TEST_COMPACT_BYTES;
        let view_ptr = view.as_bytes().as_ptr() as usize;
        assert!(view_ptr >= buffer_start && view_ptr < buffer_end);
    }

    #[test]
    fn contains_returns_false_for_absent() {
        let mp = TestMempool::new(0x1234, 42);
        let fake_hash = [0u8; 32];
        assert!(!mp.contains(&fake_hash));
    }

    #[test]
    fn confirm_by_block_acceptance_compacts() {
        let mut mp = TestMempool::new(0x1234, 42);
        let nts: [NodeTransfer; 3] = [
            sample_node_transfer(1, 1, 100),
            sample_node_transfer(2, 2, 200),
            sample_node_transfer(3, 3, 300),
        ];
        let mut hashes: [[u8; 32]; 3] = [[0u8; 32]; 3];
        for i in 0..3 {
            hashes[i] = calculate_hash(nts[i].as_bytes());
            let tx = TransactionView::from_bytes(nts[i].as_bytes()).unwrap();
            assert!(matches!(
                try_add_test_tx(&mut mp, tx, false),
                AddResult::Admitted
            ));
        }
        assert_eq!(mp.entry_count(), 3);

        // Confirm the middle one via an accepted transaction block view.
        let mut accepted_block_bytes = [0u8; HEADER_SIZE + 2 + NODE_TRANSFER_SIZE];
        let accepted_block =
            transaction_block_view(&mut accepted_block_bytes, &[nts[1].as_bytes()]);
        mp.confirm_by_block_acceptance(&accepted_block);

        assert_eq!(mp.entry_count(), 2);
        // Buffer shrank.
        let single_tx_len = nts[0].as_bytes().len() as u16;
        assert_eq!(mp.byte_usage(), single_tx_len * 2);
        // Surviving entries are present and findable.
        assert!(mp.contains(&hashes[0]));
        assert!(!mp.contains(&hashes[1]));
        assert!(mp.contains(&hashes[2]));
    }

    #[test]
    fn confirm_by_block_acceptance_compacts_multiple_gaps_and_clears_tail() {
        let mut mp = TestMempool::new(0x1234, 42);
        let nt_a = sample_node_transfer(1, 42, 100);
        let reg_b = sample_registration_with_fee(7, 2);
        let nt_c = sample_node_transfer(3, 3, 300);
        let reg_d = sample_registration_with_fee(8, 4);

        assert!(matches!(
            mp.try_add(
                TransactionView::from_bytes(nt_a.as_bytes()).unwrap(),
                11,
                false
            ),
            AddResult::Admitted
        ));
        assert!(matches!(
            mp.try_add(
                TransactionView::from_bytes(reg_b.as_bytes()).unwrap(),
                22,
                true
            ),
            AddResult::Admitted
        ));
        assert!(matches!(
            mp.try_add(
                TransactionView::from_bytes(nt_c.as_bytes()).unwrap(),
                33,
                false
            ),
            AddResult::Admitted
        ));
        assert!(matches!(
            mp.try_add(
                TransactionView::from_bytes(reg_d.as_bytes()).unwrap(),
                44,
                true
            ),
            AddResult::Admitted
        ));
        let old_byte_usage = mp.byte_usage() as usize;

        let mut accepted_block_bytes = [0u8; HEADER_SIZE + 2 + REGISTRATION_SIZE * 2];
        let accepted_block = transaction_block_view(
            &mut accepted_block_bytes,
            &[reg_b.as_bytes(), reg_d.as_bytes()],
        );
        mp.confirm_by_block_acceptance(&accepted_block);

        check_invariant(&mp);
        assert_eq!(mp.entry_count(), 2);
        assert_eq!(
            mp.byte_usage() as usize,
            nt_a.as_bytes().len() + nt_c.as_bytes().len()
        );

        let a_len = nt_a.as_bytes().len();
        let c_len = nt_c.as_bytes().len();
        assert_eq!(&mp.compact_buffer[..a_len], nt_a.as_bytes());
        assert_eq!(&mp.compact_buffer[a_len..a_len + c_len], nt_c.as_bytes());
        assert!(
            mp.compact_buffer[mp.byte_usage() as usize..old_byte_usage]
                .iter()
                .all(|byte| *byte == 0)
        );

        let entry_a = mp
            .index
            .iter()
            .find_map(|entry| {
                let entry = entry.as_ref()?;
                (entry_bytes(&mp, entry) == nt_a.as_bytes()).then_some(*entry)
            })
            .expect("surviving A entry must remain indexed");
        let entry_c = mp
            .index
            .iter()
            .find_map(|entry| {
                let entry = entry.as_ref()?;
                (entry_bytes(&mp, entry) == nt_c.as_bytes()).then_some(*entry)
            })
            .expect("surviving C entry must remain indexed");
        assert_eq!(entry_a.transaction_fee, 11);
        assert!(entry_a.is_own);
        assert!(!entry_a.is_deferred);
        assert_eq!(entry_c.transaction_fee, 33);
        assert!(!entry_c.is_own);
        assert!(!entry_c.is_deferred);
    }

    #[test]
    fn confirm_by_block_acceptance_all_removed_clears_index_and_buffer() {
        let mut mp = TestMempool::new(0x1234, 42);
        let nt_a = sample_node_transfer(1, 1, 100);
        let nt_b = sample_node_transfer(2, 2, 200);

        assert!(matches!(
            try_add_test_tx(
                &mut mp,
                TransactionView::from_bytes(nt_a.as_bytes()).unwrap(),
                false
            ),
            AddResult::Admitted
        ));
        assert!(matches!(
            try_add_test_tx(
                &mut mp,
                TransactionView::from_bytes(nt_b.as_bytes()).unwrap(),
                false
            ),
            AddResult::Admitted
        ));
        let old_byte_usage = mp.byte_usage() as usize;

        let mut accepted_block_bytes = [0u8; HEADER_SIZE + 2 + NODE_TRANSFER_SIZE * 2];
        let accepted_block = transaction_block_view(
            &mut accepted_block_bytes,
            &[nt_a.as_bytes(), nt_b.as_bytes()],
        );
        mp.confirm_by_block_acceptance(&accepted_block);

        check_invariant(&mp);
        assert_eq!(mp.entry_count(), 0);
        assert_eq!(mp.byte_usage(), 0);
        assert!(mp.index.iter().all(|entry| entry.is_none()));
        assert!(
            mp.compact_buffer[..old_byte_usage]
                .iter()
                .all(|byte| *byte == 0)
        );
    }

    #[test]
    fn contiguous_storage_invariant_after_arbitrary_ops() {
        let mut mp = TestMempool::new(0x1234, 42);
        let nt_a = sample_node_transfer(1, 1, 100);
        let nt_b = sample_node_transfer(2, 2, 200);
        let nt_c = sample_node_transfer(3, 3, 300);

        // Sequence: add A, add B, confirm A, add C
        let tx_a = TransactionView::from_bytes(nt_a.as_bytes()).unwrap();
        assert!(matches!(
            try_add_test_tx(&mut mp, tx_a, false),
            AddResult::Admitted
        ));
        check_invariant(&mp);

        let tx_b = TransactionView::from_bytes(nt_b.as_bytes()).unwrap();
        assert!(matches!(
            try_add_test_tx(&mut mp, tx_b, false),
            AddResult::Admitted
        ));
        check_invariant(&mp);

        let mut accepted_block_bytes = [0u8; HEADER_SIZE + 2 + NODE_TRANSFER_SIZE];
        let accepted_block = transaction_block_view(&mut accepted_block_bytes, &[nt_a.as_bytes()]);
        mp.confirm_by_block_acceptance(&accepted_block);
        check_invariant(&mp);

        let tx_c = TransactionView::from_bytes(nt_c.as_bytes()).unwrap();
        assert!(matches!(
            try_add_test_tx(&mut mp, tx_c, false),
            AddResult::Admitted
        ));
        check_invariant(&mp);
    }

    /// Invariant check: occupied entries form a gap-free region whose
    /// total length equals `byte_usage`.
    fn check_invariant<const C: usize, const E: usize>(mp: &Mempool<C, E>) {
        // Total bytes from index entries
        let total_bytes: u32 = mp
            .index
            .iter()
            .filter_map(|e| e.as_ref())
            .map(|e| e.length as u32)
            .sum();
        assert_eq!(total_bytes, mp.byte_usage() as u32);

        // All occupied entries' start..start+length must be inside
        // [0, byte_usage) and must not overlap.
        let mut intervals: [(u16, u16); 256] = [(0, 0); 256];
        let mut n = 0usize;
        for e in mp.index.iter().filter_map(|e| e.as_ref()) {
            intervals[n] = (e.start, e.length);
            n += 1;
        }
        // Sort by start
        for i in 1..n {
            for j in (1..=i).rev() {
                if intervals[j - 1].0 > intervals[j].0 {
                    intervals.swap(j - 1, j);
                } else {
                    break;
                }
            }
        }
        // First entry must start at 0 (gap-free from offset 0)
        if n > 0 {
            assert_eq!(intervals[0].0, 0, "first entry must start at offset 0");
        }
        // Adjacent entries must touch (no gap, no overlap)
        for i in 1..n {
            assert_eq!(
                intervals[i - 1].0 + intervals[i - 1].1,
                intervals[i].0,
                "gap or overlap between entries {} and {}",
                i - 1,
                i
            );
        }
    }

    #[test]
    fn own_classification_follows_resolved_complex_policy() {
        const OWN_NODE_ID: u32 = 7;
        let sig = [0xDD; 64];
        let hash = [0xEE; 32];

        let mut own_by_balance_input = ComplexTransaction::new(1);
        assert!(
            own_by_balance_input
                .add_balance_input(0, OWN_NODE_ID, 50, 0, &sig)
                .is_ok()
        );
        let tx = TransactionView::from_bytes(own_by_balance_input.as_bytes()).unwrap();
        assert!(is_own_transaction(&tx, OWN_NODE_ID));
        assert_eq!(transaction_expiry_sequence(&tx), 0);

        let mut own_by_balance_output = ComplexTransaction::new(1);
        assert!(own_by_balance_output.add_utxo_input(&hash, 0, &sig).is_ok());
        assert!(
            own_by_balance_output
                .add_balance_output(OWN_NODE_ID, 50)
                .is_ok()
        );
        let tx = TransactionView::from_bytes(own_by_balance_output.as_bytes()).unwrap();
        assert!(is_own_transaction(&tx, OWN_NODE_ID));
        assert_eq!(transaction_expiry_sequence(&tx), NO_EXPIRY_SEQUENCE);

        let mut zero_input_output_to_own = ComplexTransaction::new(1);
        assert!(
            zero_input_output_to_own
                .add_balance_output(OWN_NODE_ID, 50)
                .is_ok()
        );
        let tx = TransactionView::from_bytes(zero_input_output_to_own.as_bytes()).unwrap();
        assert!(!is_own_transaction(&tx, OWN_NODE_ID));
        assert_eq!(transaction_expiry_sequence(&tx), NO_EXPIRY_SEQUENCE);

        let mut utxo_only = ComplexTransaction::new(1);
        assert!(utxo_only.add_utxo_input(&hash, 0, &sig).is_ok());
        assert!(utxo_only.add_utxo_output(&hash, 50).is_ok());
        let tx = TransactionView::from_bytes(utxo_only.as_bytes()).unwrap();
        assert!(!is_own_transaction(&tx, OWN_NODE_ID));
        assert_eq!(transaction_expiry_sequence(&tx), NO_EXPIRY_SEQUENCE);
    }

    #[test]
    fn expiry_sequence_follows_anchor_dependencies() {
        let nt = sample_node_transfer_with_anchor(0, 33, 1, 100);
        let tx = TransactionView::from_bytes(nt.as_bytes()).unwrap();
        assert_eq!(transaction_expiry_sequence(&tx), 33);

        let reg = sample_registration(7);
        let tx = TransactionView::from_bytes(reg.as_bytes()).unwrap();
        assert_eq!(transaction_expiry_sequence(&tx), NO_EXPIRY_SEQUENCE);

        let sig = [0xDD; 64];
        let mut complex = ComplexTransaction::new(1);
        assert!(complex.add_balance_input(42, 1, 50, 0, &sig).is_ok());
        assert!(complex.add_balance_input(7, 2, 25, 0, &sig).is_ok());
        let tx = TransactionView::from_bytes(complex.as_bytes()).unwrap();
        assert_eq!(transaction_expiry_sequence(&tx), 7);
    }

    #[test]
    fn try_add_records_expiry_sequence() {
        let mut mp = TestMempool::new(0x1234, 42);
        let nt = sample_node_transfer_with_anchor(11, 22, 42, 100);
        assert!(matches!(
            try_add_test_tx(
                &mut mp,
                TransactionView::from_bytes(nt.as_bytes()).unwrap(),
                false
            ),
            AddResult::Admitted
        ));
        let entry = mp.index.iter().find_map(|entry| entry.as_ref()).unwrap();
        assert_eq!(entry.expiry_sequence, 22);
    }

    #[test]
    fn try_add_records_hash_crc32() {
        let mut mp = TestMempool::new(0x1234, 42);
        let nt = sample_node_transfer_with_anchor(11, 22, 42, 100);
        assert!(matches!(
            try_add_test_tx(
                &mut mp,
                TransactionView::from_bytes(nt.as_bytes()).unwrap(),
                false
            ),
            AddResult::Admitted
        ));
        let entry = mp.index.iter().find_map(|entry| entry.as_ref()).unwrap();
        assert_eq!(entry.hash_crc32, crc32_ieee(&calculate_hash(nt.as_bytes())));
        assert_ne!(entry.hash_crc32, crc32_ieee(nt.as_bytes()));
    }

    #[test]
    fn try_add_records_supplied_transaction_fee_and_top_n_returns_it() {
        let mut mp = TestMempool::new(0x1234, 42);
        let nt = sample_node_transfer_with_fee(0, 5, 100, 1);
        assert!(matches!(
            mp.try_add(
                TransactionView::from_bytes(nt.as_bytes()).unwrap(),
                77,
                false
            ),
            AddResult::Admitted
        ));

        let entry = mp.index.iter().find_map(|entry| entry.as_ref()).unwrap();
        assert_eq!(entry.transaction_fee, 77);

        // Eligibility uses the supplied transaction_fee, not the fee encoded in
        // this node-transfer test fixture (1).
        mp.recheck_eligibility(|_| 101);
        let entry = mp.index.iter().find_map(|entry| entry.as_ref()).unwrap();
        assert!(entry.is_deferred);
        mp.recheck_eligibility(|_| 177);
        let entry = mp.index.iter().find_map(|entry| entry.as_ref()).unwrap();
        assert!(!entry.is_deferred);

        let (tx, fee) = mp.top_n_for_exchange(1).next().unwrap();
        assert_eq!(tx.as_bytes(), nt.as_bytes());
        assert_eq!(fee, 77);
    }

    #[test]
    fn try_add_records_own_classification() {
        let mut mp = TestMempool::new(0x1234, 42);
        let own = sample_node_transfer(0, 42, 100);
        let non_own = sample_node_transfer(0, 7, 100);

        assert!(matches!(
            try_add_test_tx(
                &mut mp,
                TransactionView::from_bytes(own.as_bytes()).unwrap(),
                false
            ),
            AddResult::Admitted
        ));
        assert!(matches!(
            try_add_test_tx(
                &mut mp,
                TransactionView::from_bytes(non_own.as_bytes()).unwrap(),
                false
            ),
            AddResult::Admitted
        ));

        let own_count = mp
            .index
            .iter()
            .filter_map(|entry| entry.as_ref())
            .filter(|entry| entry.is_own)
            .count();
        assert_eq!(own_count, 1);
    }

    #[test]
    fn recheck_eligibility_flips_deferred_flag() {
        let mut mp = TestMempool::new(0x1234, 42);
        let nt = sample_node_transfer(0, 5, 1000); // initializer = 5, amount = 1000, fee = 1
        let tx = TransactionView::from_bytes(nt.as_bytes()).unwrap();
        assert!(matches!(
            try_add_test_tx(&mut mp, tx, false),
            AddResult::Admitted
        ));

        // Balance = 0 for everyone → tx becomes deferred.
        mp.recheck_eligibility(|_| 0u64);
        let entry = mp.index.iter().find_map(|e| e.as_ref()).unwrap();
        assert!(entry.is_deferred);

        // Balance = u64::MAX → tx becomes eligible again.
        mp.recheck_eligibility(|_| u64::MAX);
        let entry = mp.index.iter().find_map(|e| e.as_ref()).unwrap();
        assert!(!entry.is_deferred);
    }

    #[test]
    fn eligible_iter_filters_deferred() {
        let mut mp = TestMempool::new(0x1234, 42);
        let nt_a = sample_node_transfer(1, 1, 100);
        let nt_b = sample_node_transfer(2, 2, 200);
        let tx_a = TransactionView::from_bytes(nt_a.as_bytes()).unwrap();
        let tx_b = TransactionView::from_bytes(nt_b.as_bytes()).unwrap();
        assert!(matches!(
            try_add_test_tx(&mut mp, tx_a, false),
            AddResult::Admitted
        ));
        assert!(matches!(
            try_add_test_tx(&mut mp, tx_b, true),
            AddResult::Admitted
        ));

        let eligible_count = mp.eligible_iter().count();
        assert_eq!(eligible_count, 1);
    }

    #[test]
    fn top_n_for_exchange_no_alloc_no_mutation() {
        let mut mp = TestMempool::new(0x1234, 42);
        let nt = sample_node_transfer(1, 1, 100);
        for _ in 0..5 {
            let tx = TransactionView::from_bytes(nt.as_bytes()).unwrap();
            try_add_test_tx(&mut mp, tx, false);
        }
        let before = mp.entry_count();
        let yielded: usize = mp.top_n_for_exchange(3).count();
        assert!(yielded <= 3);
        assert_eq!(mp.entry_count(), before, "iterator must not mutate state");
    }

    #[test]
    fn top_n_for_exchange_orders_by_fee_per_byte() {
        let mut mp = TestMempool::new(0x1234, 42);
        let higher_rate = sample_node_transfer_with_fee(1, 7, 100, 600); // 600 / 101
        let lower_rate = sample_registration_with_fee(2, 1000); // 1000 / 189

        // Insert lower-rate first to prove the iterator is not index-order.
        assert!(matches!(
            try_add_test_tx(
                &mut mp,
                TransactionView::from_bytes(lower_rate.as_bytes()).unwrap(),
                false
            ),
            AddResult::Admitted
        ));
        assert!(matches!(
            try_add_test_tx(
                &mut mp,
                TransactionView::from_bytes(higher_rate.as_bytes()).unwrap(),
                false
            ),
            AddResult::Admitted
        ));

        let mut top = mp.top_n_for_exchange(2);
        let (tx, fee) = top.next().unwrap();
        assert_eq!(tx.as_bytes(), higher_rate.as_bytes());
        assert_eq!(fee, 600);
        let (tx, fee) = top.next().unwrap();
        assert_eq!(tx.as_bytes(), lower_rate.as_bytes());
        assert_eq!(fee, 1000);
        assert!(top.next().is_none());
    }

    #[test]
    fn top_n_for_exchange_prefers_own_on_equal_fee_per_byte() {
        let mut mp = TestMempool::new(0x1234, 42);
        let other = sample_node_transfer_with_fee(1, 7, 100, 5);
        let own = sample_node_transfer_with_fee(2, 42, 100, 5);

        // Insert other first to prove ownership is a priority tie-break.
        assert!(matches!(
            try_add_test_tx(
                &mut mp,
                TransactionView::from_bytes(other.as_bytes()).unwrap(),
                false
            ),
            AddResult::Admitted
        ));
        assert!(matches!(
            try_add_test_tx(
                &mut mp,
                TransactionView::from_bytes(own.as_bytes()).unwrap(),
                false
            ),
            AddResult::Admitted
        ));

        let mut top = mp.top_n_for_exchange(2);
        let (tx, fee) = top.next().unwrap();
        assert_eq!(tx.as_bytes(), own.as_bytes());
        assert_eq!(fee, 5);
        let (tx, fee) = top.next().unwrap();
        assert_eq!(tx.as_bytes(), other.as_bytes());
        assert_eq!(fee, 5);
    }

    #[test]
    fn top_n_for_exchange_uses_hash_crc32_tie_break() {
        let mut mp = TestMempool::new(0x1234, 42);
        let higher_crc = sample_node_transfer_with_fee(1, 7, 100, 5);
        let lower_crc = sample_node_transfer_with_fee(2, 7, 100, 5);

        // Insert higher-CRC first; force CRC values to make the ordering exact
        // and independent of accidental fixture hashes.
        assert!(matches!(
            try_add_test_tx(
                &mut mp,
                TransactionView::from_bytes(higher_crc.as_bytes()).unwrap(),
                false
            ),
            AddResult::Admitted
        ));
        assert!(matches!(
            try_add_test_tx(
                &mut mp,
                TransactionView::from_bytes(lower_crc.as_bytes()).unwrap(),
                false
            ),
            AddResult::Admitted
        ));
        set_entry_hash_crc32(&mut mp, higher_crc.as_bytes(), 20);
        set_entry_hash_crc32(&mut mp, lower_crc.as_bytes(), 10);

        let mut top = mp.top_n_for_exchange(2);
        let (tx, fee) = top.next().unwrap();
        assert_eq!(tx.as_bytes(), lower_crc.as_bytes());
        assert_eq!(fee, 5);
        let (tx, fee) = top.next().unwrap();
        assert_eq!(tx.as_bytes(), higher_crc.as_bytes());
        assert_eq!(fee, 5);
    }

    #[test]
    fn top_n_for_exchange_uses_lexicographic_bytes_on_hash_crc32_tie() {
        let mut mp = TestMempool::new(0x1234, 42);
        let lex_larger = sample_node_transfer_with_fee(2, 7, 100, 5);
        let lex_smaller = sample_node_transfer_with_fee(1, 7, 100, 5);
        assert!(lex_smaller.as_bytes() < lex_larger.as_bytes());

        // Insert lex-larger first; force a hash-CRC collision to exercise the
        // final deterministic byte tie-break.
        assert!(matches!(
            try_add_test_tx(
                &mut mp,
                TransactionView::from_bytes(lex_larger.as_bytes()).unwrap(),
                false
            ),
            AddResult::Admitted
        ));
        assert!(matches!(
            try_add_test_tx(
                &mut mp,
                TransactionView::from_bytes(lex_smaller.as_bytes()).unwrap(),
                false
            ),
            AddResult::Admitted
        ));
        set_entry_hash_crc32(&mut mp, lex_larger.as_bytes(), 0);
        set_entry_hash_crc32(&mut mp, lex_smaller.as_bytes(), 0);

        let mut top = mp.top_n_for_exchange(2);
        let (tx, fee) = top.next().unwrap();
        assert_eq!(tx.as_bytes(), lex_smaller.as_bytes());
        assert_eq!(fee, 5);
        let (tx, fee) = top.next().unwrap();
        assert_eq!(tx.as_bytes(), lex_larger.as_bytes());
        assert_eq!(fee, 5);
    }

    #[test]
    fn registration_eligibility_check() {
        let mut mp = TestMempool::new(0x1234, 42);
        let reg = sample_registration(7); // initializer = 0, registration_price = 100, fee = 1
        let tx = TransactionView::from_bytes(reg.as_bytes()).unwrap();
        assert!(matches!(
            try_add_test_tx(&mut mp, tx, false),
            AddResult::Admitted
        ));

        mp.recheck_eligibility(|_| 50u64); // 50 < 100 + 1 → deferred
        let entry = mp.index.iter().find_map(|e| e.as_ref()).unwrap();
        assert!(entry.is_deferred);

        mp.recheck_eligibility(|_| 200u64); // 200 ≥ 100 + 1 → eligible
        let entry = mp.index.iter().find_map(|e| e.as_ref()).unwrap();
        assert!(!entry.is_deferred);
    }
}
