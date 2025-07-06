
import pandas as pd
from Bio import SeqIO
import re
from pathlib import Path

def parse_amino_acid_change(protein_change):
    # Only taken the first variant listed
    first = protein_change.split(',')[0].strip()
    match = re.match(r"([A-Z])(\d+)([A-Z])", first)
    if match:
        orig, pos, new = match.groups()
        return int(pos), orig, new
    return None

def main():
    csv_path = "BRCA1_Variant_Data.csv"
    fasta_path = "BRCA1.fasta"
    output_path = "mutants.csv"

    df = pd.read_csv(csv_path)
    print(f"Total variants loaded: {len(df)}")

    #
    brca_seq = str(SeqIO.read(fasta_path, "fasta").seq)
    print(f"Loaded BRCA1 sequence: {len(brca_seq)} amino acids")

    variants = []
    for _, row in df.iterrows():
        prot_change = str(row.get("Protein change", ""))
        result = parse_amino_acid_change(prot_change)
        if result is None:
            continue
        pos, orig, new = result
        if pos <= len(brca_seq) and brca_seq[pos - 1] == orig:
            variants.append((pos, orig, new))

    print(f"Valid variants found: {len(variants)}")

    if not variants:
        print("No valid mutant sequences generated.")
        return

    
    pd.DataFrame(variants, columns=["Position", "Original", "Mutant"]).to_csv(output_path, index=False)
    print(f"Saved valid mutations to {output_path}")


if __name__ == "__main__":
    main()
