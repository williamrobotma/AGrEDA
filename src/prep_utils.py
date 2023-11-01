import itertools

cell_subclass_to_spot_composition = {
    "Astro": {"Astrocytes deep", "Astrocytes superficial"},
    "CR": set(),
    "Batch Grouping": {"Excitatory layer 5/6"},  # all cell clusters are L5
    "L2/3 IT": {"Excitatory layer II", "Excitatory layer 3"},
    "L4": {"Excitatory layer 4"},
    "L5 PT": {"Excitatory layer 5/6"},
    "L5 IT": {"Excitatory layer 5/6"},
    "L6 CT": {"Excitatory layer 5/6"},
    "L6 IT": {"Excitatory layer 5/6"},
    "L6b": {"Excitatory layer 5/6"},
    "NP": {"Excitatory layer 5/6"},  # all NP are L5 or L6
    "Endo": {"Endothelial", "Choroid plexus"},
    "High Intronic": {"Excitatory layer 5/6"},  # all High Intronic are VISp L5 Endou
    ## Doublets; these are cell clusters
    "Peri": {"Endothelial", "Choroid plexus"},  # pericyte
    "SMC": set(),  # smooth muscle cell
    "VLMC": set(),  # vascular leptomeningeal cell
    "Macrophage": {"Microglia"},
    "Lamp5": {
        "Interneurons",
        "Interneurons deep",
    },  # "We define six subclasses of GABAergic cells: Sst, Pvalb, Vip, Lamp5, Sncg and Serpinf1, and two distinct types: Sst–Chodl and Meis2–Adamts19 (Fig. 1c). We represent the taxonomy by constellation diagrams, dendrograms, layer-of-isolation, and the expression of select marker genes (Fig. 5a–f). The major division among GABAergic types largely corresponds to their developmental origin in the medial ganglionic eminence (Pvalb and Sst subclasses) or caudal ganglionic eminence (Lamp5, Sncg, Serpinf1 and Vip subclasses)."
    "Meis2": {"Interneurons", "Interneurons deep"},
    "Pvalb": {"Interneurons", "Interneurons deep"},
    "Serpinf1": {"Interneurons", "Interneurons deep"},
    "Sncg": {"Interneurons", "Interneurons deep"},
    "Sst": {"Interneurons", "Interneurons deep"},
    "Vip": {"Interneurons", "Interneurons deep"},
    "Oligo": {"Oligodendrocytes", "OPC"},
    "keep_the_rest": {"Ependymal", "NSC", "Neural progenitors", "Neuroblasts"},
}

cell_cluster_cell_type_to_spot_composition = {
    "Doublet VISp L5 NP and L6 CT": {"Excitatory layer 5/6"},
    "Doublet Endo": {
        "Endothelial",
        "Choroid plexus",
    },  # no choroid plexus in other dataset
    "Doublet Astro Aqp4 Ex": {"Astrocytes deep", "Astrocytes superficial"},
}

cell_cluster_to_cell_type = {
    "Doublet VISp L5 NP and L6 CT": "Doublet VISp L5 NP and L6 CT",
    "Doublet Endo and Peri_1": "Doublet Endo",
    "Doublet Astro Aqp4 Ex": "Doublet Astro Aqp4 Ex",
    "Doublet SMC and Glutamatergic": "Doublet Endo",
    "Doublet Endo Peri SMC": "Doublet Endo",
}

def get_st_sub_map():
    st_sub_map = {}
    for k, s in itertools.chain(
        cell_cluster_cell_type_to_spot_composition.items(),
        cell_subclass_to_spot_composition.items(),
    ):
        if s is not None:
            if k == "keep_the_rest":
                for s_ in s:
                    st_sub_map[s_] = [s_]
            elif len(s) > 0:
                names = sorted(list(s))
                st_sub_map["/".join(names)] = names
    return st_sub_map