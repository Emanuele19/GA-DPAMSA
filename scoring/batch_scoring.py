import torch


def compute_sp_score(
        aligned_seqs: torch.Tensor,
        match_reward: float = 4.0,
        mismatch_penalty: float = -4.0,
        gap_penalty: float = -2.0,
        gap_token: int = 5,
        pad_token: int = 0
) -> torch.Tensor:
    """
    Calculates the Sum-of-Pairs (SP) score for a batch of alignments using GPU Broadcasting.

    This function replaces nested CPU loops with tensor expansion, allowing for
    massive parallelism. It compares every row against every other row simultaneously.

    Mechanism:
        1. Create two views of the batch: (Batch, Rows, 1, Len) and (Batch, 1, Rows, Len).
        2. Comparing them generates a (Batch, Rows, Rows, Len) tensor representing
           the pairwise relationship of every nucleotide.
        3. Apply masks to handle Gaps and Technical Padding.
        4. Sum the scores over the upper triangular matrix to count unique pairs.

    Args:
        aligned_seqs: (Batch, Rows, Len) Integer Tensor containing the aligned sequences.
        match_reward: Score added for identical non-gap characters (default: 4.0).
        mismatch_penalty: Score added (usually negative) for different non-gap characters (default: -4.0).
        gap_penalty: Score added (usually negative) when a residue is aligned with a gap (default: -2.0).
        gap_token: The integer ID representing a Gap (default: 5).
        pad_token: The integer ID representing Technical Padding (default: 0).

    Returns:
        scores: (Batch,) Float Tensor containing the total SP score for each alignment in the batch.
    """
    batch_size, num_rows, length = aligned_seqs.shape
    device = aligned_seqs.device

    # =========================================================================
    # 1. BROADCASTING SETUP
    # =========================================================================
    # We want to compare every row 'i' with every row 'j'.
    # PyTorch broadcasting automatically expands dimensions 1 and 2 to match.

    # View A: Shape (Batch, Rows, 1, Len) -> Represents Row 'i'
    seqs_i = aligned_seqs.unsqueeze(2)

    # View B: Shape (Batch, 1, Rows, Len) -> Represents Row 'j'
    seqs_j = aligned_seqs.unsqueeze(1)

    # =========================================================================
    # 2. BOOLEAN MASKS (Identify Token Types)
    # =========================================================================
    # Resulting Shape for all masks: (Batch, Rows, Rows, Len)

    is_gap_i = (seqs_i == gap_token)
    is_gap_j = (seqs_j == gap_token)

    is_pad_i = (seqs_i == pad_token)
    is_pad_j = (seqs_j == pad_token)

    # VALIDITY MASK:
    # We must strictly ignore any comparison that involves a technical PAD token.
    # If the network output includes padding (e.g. at the end of the tensor),
    # it must not influence the biological score.
    valid_comparison_mask = ~(is_pad_i | is_pad_j)

    # =========================================================================
    # 3. SCORING LOGIC
    # =========================================================================

    # A. MATCH: Characters are identical, and NEITHER is a gap.
    #    (Note: 'Gap == Gap' is handled separately, usually score 0).
    matches = (seqs_i == seqs_j) & (~is_gap_i) & (~is_gap_j)

    # B. MISMATCH: Characters are different, and NEITHER is a gap.
    mismatches = (seqs_i != seqs_j) & (~is_gap_i) & (~is_gap_j)

    # C. GAP PENALTY: Exactly one of the two characters is a gap (XOR operation).
    #    Case: (Residue vs Gap) or (Gap vs Residue).
    gaps = (is_gap_i ^ is_gap_j)

    # =========================================================================
    # 4. WEIGHTED SUMMATION
    # =========================================================================

    # Apply weights. We convert booleans to float (True=1.0, False=0.0).
    # Note: Gap-vs-Gap (is_gap_i & is_gap_j) results in 0.0 here, which is standard.
    scores_tensor = (
            matches.float() * match_reward +
            mismatches.float() * mismatch_penalty +
            gaps.float() * gap_penalty
    )

    # Zero out invalid scores (those coming from padding)
    scores_tensor = scores_tensor * valid_comparison_mask.float()

    # =========================================================================
    # 5. AGGREGATION
    # =========================================================================

    # Sum over the Length dimension first to get the total score for each pair of rows.
    # Shape becomes: (Batch, Rows, Rows)
    pairwise_matrix = scores_tensor.sum(dim=3)

    # We only want to sum unique pairs (Row i vs Row j where i < j).
    # We ignore the diagonal (Row i vs Row i) and the lower triangle (duplicates).

    # Create an upper-triangle mask (diagonal=1 excludes the main diagonal).
    triu_mask = torch.ones((num_rows, num_rows), device=device).triu(diagonal=1)

    # Apply the mask. The mask broadcasts over the Batch dimension automatically.
    # Sum over dimensions 1 and 2 (Rows x Rows) to get a single scalar per batch item.
    final_scores = (pairwise_matrix * triu_mask).sum(dim=(1, 2))

    return final_scores


def compute_cs_score(
        aligned_seqs: torch.Tensor,
        gap_token: int = 5,
        pad_token: int = 0
) -> torch.Tensor:
    """
    Calculates the Column Score (CS) percentage for a batch on GPU.

    Definition:
        CS is the percentage of columns in the alignment that are "fully conserved".
        A column is conserved if:
        1. All characters in the column are identical.
        2. The column does NOT contain a Gap.
        3. The column does NOT contain technical Padding.

    Args:
        aligned_seqs: (Batch, Rows, Len) Integer Tensor.
        gap_token: Integer ID for Gap.
        pad_token: Integer ID for Padding.

    Returns:
        cs_percent: (Batch,) Float Tensor values in range [0.0, 100.0].
    """
    batch_size, num_rows, length = aligned_seqs.shape

    # =========================================================================
    # 1. CHECK IDENTITY
    # =========================================================================
    # We compare all rows against the first row (Reference Row).

    # Slice to keep dimensions: (Batch, 1, Len)
    ref_row = aligned_seqs[:, 0:1, :]

    # Compare: (Batch, Rows, Len). Returns True where row[i] matches row[0].
    matches_ref = (aligned_seqs == ref_row)

    # Collapse Rows: A column is identical if .all() rows match the reference.
    # Shape: (Batch, Len)
    all_identical = matches_ref.all(dim=1)

    # =========================================================================
    # 2. CHECK INVALID TOKENS (Gaps & Pads)
    # =========================================================================

    # Check if a column contains ANY gap across the rows.
    # Shape: (Batch, Len)
    has_gaps = (aligned_seqs == gap_token).any(dim=1)

    # Check if a column contains ANY padding across the rows.
    # Shape: (Batch, Len)
    has_pads = (aligned_seqs == pad_token).any(dim=1)

    # =========================================================================
    # 3. COMPUTE PERCENTAGE
    # =========================================================================

    # A column is conserved if it is Identical AND has No Gaps AND has No Pads.
    conserved_cols = all_identical & (~has_gaps) & (~has_pads)

    # Count conserved columns per batch item.
    num_conserved = conserved_cols.float().sum(dim=1)

    # Normalize by total length.
    # We add a small epsilon (1e-8) to avoid division by zero if length is 0.
    cs_percent = (num_conserved / (length + 1e-8)) * 100.0

    return cs_percent