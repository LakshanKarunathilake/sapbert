#!/usr/bin/env python3
"""
SapBERT Training Data Generation Script
Extracted from generate_pretraining_data.ipynb

This script processes UMLS data to generate positive pairs for SapBERT training.
"""

from tqdm import tqdm
import itertools
import random
import os

def main():
    print("Starting SapBERT training data generation...")
    
    # Step 1: Load MRCONSO.RRF and basic preprocessing
    print("\n=== Step 1: Loading MRCONSO.RRF ===")
    mrconso_path = "2025AA/META/MRCONSO.RRF"
    
    if not os.path.exists(mrconso_path):
        print(f"Error: {mrconso_path} not found!")
        return
    
    with open(mrconso_path, "r") as f:
        lines = f.readlines()
    print(f"Loaded {len(lines)} lines from MRCONSO.RRF")
    
    # Step 2: Use only English names
    print("\n=== Step 2: Filtering English names ===")
    cleaned = []
    for l in tqdm(lines):
        lst = l.rstrip("\n").split("|")
        cui, lang, synonym = lst[0], lst[1], lst[14]
        if lang != "ENG": 
            continue  # comment this out if you need all languages
        row = cui + "||" + synonym.lower()
        cleaned.append(row)
    print(f"Found {len(cleaned)} English entries")
    
    # Step 3: Remove duplicates
    print("\n=== Step 3: Removing duplicates ===")
    print(f"Before deduplication: {len(cleaned)}")
    cleaned = list(set(cleaned))
    print(f"After deduplication: {len(cleaned)}")
    
    # Step 4: Add tradenames (optional)
    print("\n=== Step 4: Adding tradenames ===")
    mrrel_path = "UMLS/2025AA/META/MRREL.RRF"
    
    if os.path.exists(mrrel_path):
        print("Loading MRREL.RRF for tradename processing...")
        with open(mrrel_path, "r") as f:
            lines = f.readlines()
        print(f"Loaded {len(lines)} lines from MRREL.RRF")
        
        # Build umls_dict from cleaned data
        print("Building UMLS dictionary...")
        umls_dict = {}
        for line in tqdm(cleaned):
            try:
                parts = line.split("||")
                if len(parts) != 2:
                    continue
                cui, name = parts[0], parts[1]
                if cui in umls_dict:
                    umls_dict[cui].append(name)
                else:
                    umls_dict[cui] = [name]
            except Exception as e:
                continue
        
        print(f"Built umls_dict with {len(umls_dict)} CUIs")
        
        # Process tradename mappings
        print("Processing tradename mappings...")
        tradename_mappings = {}
        for l in tqdm(lines):
            if "has_tradename" in l or "tradename_of" in l:
                cells = l.split("|")
                head, tail = cells[0], cells[4]
                try:  # if in CUI
                    sfs = umls_dict[tail]
                    tradename_mappings[head] = sfs
                except:
                    continue
        print(f"Found {len(tradename_mappings)} tradename mappings")
        
        # Add tradenames to cleaned data
        print(f"Before adding tradenames: {len(cleaned)}")
        for cui, synonyms in tradename_mappings.items():
            for s in synonyms:
                row = cui + "||" + s.lower()
                cleaned.append(row)
        print(f"After adding tradenames: {len(cleaned)}")
        
        # Remove duplicates again
        print("Removing duplicates after tradename addition...")
        cleaned = list(set(cleaned))
        print(f"Final cleaned data size: {len(cleaned)}")
    else:
        print(f"Warning: {mrrel_path} not found, skipping tradename processing")
    
    # Step 5: Generate positive pairs
    print("\n=== Step 5: Generating positive pairs ===")
    
    # Rebuild umls_dict from final cleaned data
    print("Rebuilding UMLS dictionary for pair generation...")
    umls_dict = {}
    for line in tqdm(cleaned):
        try:
            cui, name = line.split("||")
            if cui in umls_dict:
                umls_dict[cui].append(name)
            else:
                umls_dict[cui] = [name]
        except:
            continue
    
    def gen_pairs(input_list):
        """Generate all possible pairs from input list"""
        return list(itertools.combinations(input_list, r=2))
    
    # Test the function
    test_pairs = gen_pairs([1, 2, 3])
    print(f"Test pairs: {test_pairs}")
    
    # Generate positive pairs
    print("Generating positive pairs...")
    pos_pairs = []
    for k, v in tqdm(umls_dict.items()):
        pairs = gen_pairs(v)
        if len(pairs) > 50:  # if >50 pairs, then trim to 50 pairs
            pairs = random.sample(pairs, 50)
        for p in pairs:
            line = str(k) + "||" + p[0] + "||" + p[1]
            pos_pairs.append(line)
    
    print(f"Generated {len(pos_pairs)} positive pairs")
    
    # Show sample pairs
    print("Sample positive pairs:")
    for i, pair in enumerate(pos_pairs[:3]):
        print(f"  {i+1}: {pair}")
    
    # Step 6: Save the training file
    print("\n=== Step 6: Saving training file ===")
    output_file = './training_file_umls2025aa_en_uncased_no_dup_pairwise_pair_th50.txt'
    
    with open(output_file, 'w') as f:
        for line in pos_pairs:
            f.write("%s\n" % line)
    
    print(f"Training data saved to: {output_file}")
    print(f"Total positive pairs: {len(pos_pairs)}")
    print("Training data generation completed successfully!")

if __name__ == "__main__":
    main()
