from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mygene

MG = mygene.MyGeneInfo()


@dataclass
class GeneMappingOutput:
    """Gene mapping results data structure.

    Attributes:
        genes: List of original gene ids.
        mapping_full: Dictionary mapping from query gene id to target ids as
            a list. If the mapping is not available, then this gene id would
            not appear in this mapping. Thus, this dictionary mmight contain
            less elements than the total number of the queried genes.
        mapping_combined: Dictionary mapping from query gene id to combined
            target ids (concatenated by "|", e.g., ["id1", "id2"] would be
            converted to "id1|id2"). If the gene id conversion is not
            applicable, then we will map it to "N/A". Thus, this dictionary
            contains the same number of element as the total number of the
            queried genes.
        mapping_reduced: Similar to mapping_combined, but only use the first
            mapped ids (sorted alphabetically) when multiple ids are available.
            Furthermore, use the query gene id as the target gene id if the
            mapping is unavailable. This dictionary contains the same number of
            elements as the total number of the queried genes.

    """
    genes: List[str]
    mapping_full: Dict[str, List[str]]
    mapping_combined: Dict[str, str] = field(init=False)
    mapping_reduced: Dict[str, str] = field(init=False)

    def __post_init__(self):
        self.mapping_combined = {}
        self.mapping_reduced = {}
        for g in self.genes:
            if (ensembl := self.mapping_full.get(g)) is None:
                self.mapping_combined[g] = "N/A"
                self.mapping_reduced[g] = g
            else:
                ensembl = sorted(set(ensembl))
                self.mapping_full[g] = ensembl
                self.mapping_reduced[g] = ensembl[0]
                self.mapping_combined[g] = "|".join(ensembl)


def symbol_to_ensembl(
    genes: List[str],
    species: str = "human",
    extra_query_kwargs: Optional[Dict[str, Any]] = None,
) -> GeneMappingOutput:
    # Query from MyGene.Info server
    print(f"Querying {len(genes):,} genes")
    query_results = MG.querymany(
        genes,
        species=species,
        scopes="symbol",
        fields="ensembl.gene",
        **(extra_query_kwargs or {}),
    )

    # Unpack query results
    symbol_to_ensembl_dict = {}
    for res in query_results:
        symbol = res["query"]
        if (ensembl := res.get("ensembl")) is None:
            continue

        if isinstance(ensembl, dict):
            new_ensembl_genes = [ensembl["gene"]]
        elif isinstance(ensembl, list):
            new_ensembl_genes = [i["gene"] for i in ensembl]
        else:
            raise ValueError(f"Unknown ensembl query result type {type(ensembl)}: {ensembl!r}")

        symbol_to_ensembl_dict[symbol] = symbol_to_ensembl_dict.get(symbol, []) + new_ensembl_genes

    print(
        f"Successfully mapped {len(symbol_to_ensembl_dict):,} out of "
        f"{len(genes):,} genes ({len(symbol_to_ensembl_dict) / len(genes):.1%})",
    )

    return GeneMappingOutput(genes, symbol_to_ensembl_dict)


if __name__ == "__main__":
    species = "human"
    gene_table_symbol_to_ensembl = {
        "PPIEL": ["ENSG00000243970", "ENSG00000291129"],
        "PRDM16": ["ENSG00000142611"],
        "PEX10": ["ENSG00000157911"],
        "RNA5-8SN5": ["ENSG00000274917"],
        "DOESNOTEXIST": None,
    }
    genes = list(gene_table_symbol_to_ensembl)

    res = symbol_to_ensembl(genes, species=species)
    print(res.mapping_full)
    assert res.genes == genes
    for g in genes:
        if gene_table_symbol_to_ensembl[g] is None:
            assert g not in res.mapping_full
            assert res.mapping_combined[g] == "N/A"
            assert res.mapping_reduced[g] == g
        else:
            assert res.mapping_full[g] == gene_table_symbol_to_ensembl[g]
            assert res.mapping_combined[g] == "|".join(gene_table_symbol_to_ensembl[g])
            assert res.mapping_reduced[g] == gene_table_symbol_to_ensembl[g][0]
