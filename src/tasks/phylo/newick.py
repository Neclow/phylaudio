from ete3 import Tree


def apply_language_mapping_to_newick(
    newick_file, languages, key="full", output_file=None
):
    tree = Tree(newick_file, format=0)

    for leaf in tree.iter_leaves():
        leaf.name = languages[leaf.name][key]

    return tree.write(outfile=output_file)
