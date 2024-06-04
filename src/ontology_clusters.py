def load_facets(facets_file):
    facets = {}
    with open(facets_file, 'r') as file:
        current_facet = None
        for line in file:
            if line.startswith("Facet"):
                current_facet = line.strip()
                facets[current_facet] = {}
            elif line.startswith("Cluster"):
                _, concepts = line.split(':')
                concepts = concepts.strip().split(', ')
                for concept in concepts:
                    if concept not in facets[current_facet]:
                        facets[current_facet][concept] = []
                    facets[current_facet][concept].append(line.split(':')[1].strip())
    return facets

def find_concept_clusters(concept, facets):
    concept_clusters = {}
    for facet, concepts in facets.items():
        if concept in concepts:
            concept_clusters[facet] = ', '.join(concepts[concept])
    return concept_clusters

def process_concepts(types_file, facets, output_file):
    with open(types_file, 'r') as types, open(output_file, 'w') as out_file:
        for concept in types:
            concept = concept.strip()
            concept_clusters = find_concept_clusters(concept, facets)
            out_file.write(f'{concept}:\n')
            for facet, clusters in concept_clusters.items():
                out_file.write(f'  {facet} {clusters}\n')
            out_file.write('\n')

def main(facets_file, types_file, output_file):
    facets = load_facets(facets_file)
    process_concepts(types_file, facets, output_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Match concepts with clusters from different facets.")
    parser.add_argument('--facets', type=str, required=True, help='Path to input file with facets and clusters')
    parser.add_argument('--types', type=str, required=True, help='Path to input file with types (concepts)')
    parser.add_argument('--output', type=str, required=True, help='Path to output file')

    args = parser.parse_args()
    main(args.facets, args.types, args.output)


