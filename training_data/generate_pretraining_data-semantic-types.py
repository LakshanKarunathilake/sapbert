#!/usr/bin/env python3
"""
SapBERT Training Data Generation Script (+ semantic types once + TUI index files)

Output line format (types only once at the end):
CUI||alias1||alias2||t1|t2|...

Also writes:
- tui2idx.json, idx2tui.json
- tui2idx.tsv,  idx2tui.tsv
"""

from tqdm import tqdm
import itertools
import random
import os
import json

# -----------------------------
# Config
# -----------------------------
UMLS_ROOT = "2025AA/META"
MRCONSO = os.path.join(UMLS_ROOT, "MRCONSO.RRF")
MRREL   = os.path.join(UMLS_ROOT, "MRREL.RRF")
MRSTY   = os.path.join(UMLS_ROOT, "MRSTY.RRF")

LANG_FILTER = {"ENG"}      # keep only these languages (set to None to keep all)
LOWERCASE_SYNONYMS = True  # lowercase alias strings
PAIR_CAP_PER_CUI = 50      # max pairs per CUI (randomly sampled if exceeded)
USE_STY_NAME = False       # False => use TUI codes; True => use STY names
RANDOM_SEED = 13           # for reproducible sampling
INDEX_START = 0            # starting index for tui2idx mapping (set to 1 if preferred)

# Output filenames
def out_pairs_name():
    ext = "styname" if USE_STY_NAME else "tui"
    return f'./training_file_umls2025aa_en_uncased_no_dup_pairwise_pair_th{PAIR_CAP_PER_CUI}_{ext}.txt'

TUI2IDX_JSON = "./tui2idx.json"
IDX2TUI_JSON = "./idx2tui.json"
TUI2IDX_TSV  = "./tui2idx.tsv"
IDX2TUI_TSV  = "./idx2tui.tsv"

# -----------------------------
# Helpers
# -----------------------------
def load_mrconso_keep_langs(path, lang_filter=None, lowercase=True):
    """Return list of 'CUI||alias' strings filtered by language."""
    if not os.path.exists(path):
        print(f"Error: {path} not found!")
        return None

    print(f"Loading MRCONSO from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Loaded {len(lines)} lines from MRCONSO.RRF")

    cleaned = []
    print("Filtering entries...")
    for l in tqdm(lines):
        lst = l.rstrip("\n").split("|")
        # MRCONSO columns (subset): CUI|LAT|...|STR|...
        cui, lat, synonym = lst[0], lst[1], lst[14]
        if (lang_filter is not None) and (lat not in lang_filter):
            continue
        if lowercase:
            synonym = synonym.lower()
        cleaned.append(cui + "||" + synonym)
    print(f"Found {len(cleaned)} entries after language filtering")

    # Deduplicate
    print("Removing duplicates...")
    before = len(cleaned)
    cleaned = list(set(cleaned))
    print(f"Before: {before} | After: {len(cleaned)}")

    return cleaned

def add_tradenames_from_mrrel(cleaned, mrrel_path):
    """Extend cleaned 'CUI||alias' list using MRREL has_tradename/tradename_of."""
    if not os.path.exists(mrrel_path):
        print(f"Warning: {mrrel_path} not found, skipping tradename processing")
        return cleaned

    print("\n=== Step: Adding tradenames from MRREL ===")
    print(f"Loading MRREL from: {mrrel_path}")
    with open(mrrel_path, "r", encoding="utf-8") as f:
        rel_lines = f.readlines()
    print(f"Loaded {len(rel_lines)} lines from MRREL.RRF")

    # Build CUI -> [aliases] dictionary from current cleaned list
    print("Building UMLS dictionary from cleaned synonyms...")
    umls_dict = {}
    for line in tqdm(cleaned):
        parts = line.split("||")
        if len(parts) != 2:
            continue
        cui, name = parts[0], parts[1]
        umls_dict.setdefault(cui, []).append(name)

    # Process tradename mappings: when RELA contains has_tradename or tradename_of
    print("Processing tradename relationships...")
    tradename_mappings = {}  # head CUI -> list of tradename strings (from tail CUI)
    for l in tqdm(rel_lines):
        # MRREL columns (subset): CUI1|AUI1|STYPE1|REL|CUI2|AUI2|STYPE2|RELA|...
        cells = l.split("|")
        if len(cells) < 9:
            continue
        rela = cells[7]
        if rela not in ("has_tradename", "tradename_of"):
            continue
        head, tail = cells[0], cells[4]
        if tail in umls_dict:
            tradename_mappings[head] = umls_dict[tail]

    print(f"Found {len(tradename_mappings)} tradename mappings")
    print(f"Before adding tradenames: {len(cleaned)}")
    for cui, synonyms in tradename_mappings.items():
        for s in synonyms:
            cleaned.append(cui + "||" + s)
    # Deduplicate again
    cleaned = list(set(cleaned))
    print(f"After adding tradenames (deduped): {len(cleaned)}")
    return cleaned

def build_umls_dict(cleaned):
    """Build CUI -> [aliases] dictionary from 'CUI||alias' strings."""
    print("Rebuilding UMLS dictionary from final cleaned data...")
    umls_dict = {}
    for line in tqdm(cleaned):
        try:
            cui, name = line.split("||")
            umls_dict.setdefault(cui, []).append(name)
        except ValueError:
            continue
    return umls_dict

def load_mrsty_types(mrsty_path, use_sty_name=False):
    """
    Return dict: CUI -> sorted list of semantic type strings.
    If use_sty_name=False, use TUI (e.g., T047). If True, use STY name.
    """
    print("\n=== Step: Loading semantic types from MRSTY ===")
    if not os.path.exists(mrsty_path):
        print(f"Warning: {mrsty_path} not found. Semantic types will be empty.")
        return {}

    print(f"Loading MRSTY from: {mrsty_path}")
    with open(mrsty_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Loaded {len(lines)} lines from MRSTY.RRF")

    cui2types = {}
    for l in tqdm(lines):
        # MRSTY columns: CUI|TUI|STN|STY|ATUI|CVF
        cells = l.rstrip("\n").split("|")
        if len(cells) < 4:
            continue
        cui, tui, sty = cells[0], cells[1], cells[3]
        key = sty if use_sty_name else tui
        if not key:
            continue
        cui2types.setdefault(cui, set()).add(key)

    # convert sets to sorted lists for deterministic output
    for cui in list(cui2types.keys()):
        cui2types[cui] = sorted(cui2types[cui])

    print(f"Built semantic type map for {len(cui2types)} CUIs")
    return cui2types

def gen_pairs(input_list):
    """Generate all possible unordered pairs from input list."""
    return list(itertools.combinations(input_list, r=2))

def write_tui_index_files(cui2types, index_start=0):
    """
    Build and write TUI index maps from TUIs present for the CUIs in cui2types.
    Writes JSON and TSV for both directions.
    """
    if not os.path.exists(MRSTY):
        print(f"Warning: {MRSTY} not found. Skipping TUI index generation.")
        return

    dataset_cuis = set(cui2types.keys())
    tui_set = set()
    with open(MRSTY, "r", encoding="utf-8") as f:
        for l in f:
            cells = l.rstrip("\n").split("|")
            if len(cells) < 2:
                continue
            cui, tui = cells[0], cells[1]
            if cui in dataset_cuis and tui:
                tui_set.add(tui)

    tui_list = sorted(tui_set)  # deterministic order
    tui2idx = {tui: i + index_start for i, tui in enumerate(tui_list)}
    idx2tui = {str(i + index_start): tui for i, tui in enumerate(tui_list)}

    print(f"TUI index size: {len(tui2idx)}")

    # JSON
    with open(TUI2IDX_JSON, "w", encoding="utf-8") as f:
        json.dump(tui2idx, f, ensure_ascii=False, indent=2)
    with open(IDX2TUI_JSON, "w", encoding="utf-8") as f:
        json.dump(idx2tui, f, ensure_ascii=False, indent=2)

    # TSV
    with open(TUI2IDX_TSV, "w", encoding="utf-8") as f:
        for tui, idx in tui2idx.items():
            f.write(f"{tui}\t{idx}\n")
    with open(IDX2TUI_TSV, "w", encoding="utf-8") as f:
        for idx, tui in idx2tui.items():
            f.write(f"{idx}\t{tui}\n")

    print(f"Wrote: {TUI2IDX_JSON}, {IDX2TUI_JSON}, {TUI2IDX_TSV}, {IDX2TUI_TSV}")

# -----------------------------
# Main
# -----------------------------
def main():
    print("Starting SapBERT training data generation (semantic types once) + TUI indices...\n")
    random.seed(RANDOM_SEED)

    # Step 1: Load MRCONSO and preprocess
    print("=== Step 1: Loading MRCONSO.RRF ===")
    cleaned = load_mrconso_keep_langs(MRCONSO, lang_filter=LANG_FILTER, lowercase=LOWERCASE_SYNONYMS)
    if cleaned is None:
        return

    # Step 2: Optional tradenames
    cleaned = add_tradenames_from_mrrel(cleaned, MRREL)

    # Step 3: Load MRSTY semantic types
    cui2types = load_mrsty_types(MRSTY, use_sty_name=USE_STY_NAME)

    # Step 4: Build CUI -> aliases dictionary
    umls_dict = build_umls_dict(cleaned)
    print(f"Final CUIs in dictionary: {len(umls_dict)}")

    # Step 5: Generate positive pairs with semantic types (types ONCE)
    print("\n=== Step 5: Generating positive pairs with semantic types ===")
    test_pairs = gen_pairs([1, 2, 3])
    print(f"Test pairs (sanity check): {test_pairs}")

    print("Generating...")
    pos_pairs = []
    for cui, aliases in tqdm(umls_dict.items()):
        aliases = list(dict.fromkeys(aliases))  # stable unique within CUI
        pairs = gen_pairs(aliases)
        if PAIR_CAP_PER_CUI is not None and len(pairs) > PAIR_CAP_PER_CUI:
            pairs = random.sample(pairs, PAIR_CAP_PER_CUI)

        # Types once per CUI
        types = cui2types.get(cui, [])
        types_str = "|".join(types) if types else ""

        for a1, a2 in pairs:
            # NOTE: types only once at the end now
            line = f"{cui}||{a1}||{a2}||{types_str}"
            pos_pairs.append(line)

    print(f"Generated {len(pos_pairs)} positive pairs")

    # Show sample
    print("Sample positive pairs:")
    for i, pair in enumerate(pos_pairs[:3]):
        print(f"  {i+1}: {pair}")

    # Step 6: Save pairs
    print("\n=== Step 6: Saving training file ===")
    output_file = out_pairs_name()
    with open(output_file, "w", encoding="utf-8") as f:
        for line in pos_pairs:
            f.write(line + "\n")
    print(f"Training data saved to: {output_file}")
    print(f"Total positive pairs: {len(pos_pairs)}")

    # Step 7: Build and save TUI index maps
    print("\n=== Step 7: Building TUI index maps ===")
    write_tui_index_files(cui2types, index_start=INDEX_START)

    print("Training data generation completed successfully!")

if __name__ == "__main__":
    main()
