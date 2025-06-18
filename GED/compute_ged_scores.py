import argparse
import numpy as np
import pickle
from ged import graph_edit_distance
import dgl
from scipy.spatial import distance
from tqdm import tqdm

def compute_ged_costs(g1, g2, psg_cat):
    """
    g1, g2: NetworkX Graphs
    """
    g1_emb = sorted(list(g1.nodes.data("embedding")), key=lambda x: x[0])
    g2_emb = sorted(list(g2.nodes.data("embedding")), key=lambda x: x[0])

    g1_ins_del = np.ones((len(g1_emb),), dtype=np.float64)
    g2_ins_del = np.ones((len(g2_emb),), dtype=np.float64)
        

    g1_emb_matrix = np.vstack([emb for idx, emb in g1_emb])
    g2_emb_matrix = np.vstack([emb for idx, emb in g2_emb])

    g1_ins_del = np.squeeze(distance.cdist( g1_emb_matrix, np.array([psg_cat[8][-2]]), 'cosine'))
    g2_ins_del = np.squeeze(distance.cdist( g2_emb_matrix, np.array([psg_cat[8][-2]]), 'cosine'))                            
    substitution = distance.cdist( g1_emb_matrix, g2_emb_matrix, 'cosine')
            
    return (substitution), (g1_ins_del), (g2_ins_del)


def main():
    parser = argparse.ArgumentParser(description='Compute Graph Edit Distance scores')
    parser.add_argument('--test_dataset', required=True, help='Path to test dataset pickle file')
    parser.add_argument('--psg_cat', required=True, help='Path to PSG category embeddings pickle file')
    parser.add_argument('--output', required=True, help='Path to save output ground truth pickle file')
    args = parser.parse_args()

    # Validate that input files are pickle files
    if not args.test_dataset.endswith('.pkl'):
        raise ValueError(f"Test dataset file must be a pickle file (.pkl): {args.test_dataset}")
    if not args.psg_cat.endswith('.pkl'):
        raise ValueError(f"PSG category file must be a pickle file (.pkl): {args.psg_cat}")
    if not args.output.endswith('.pkl'):
        raise ValueError(f"Output file must be a pickle file (.pkl): {args.output}")

    with open(args.test_dataset, "rb") as f:
        test_dataset = pickle.load(f)
    print(f"Test dataset loaded from: {args.test_dataset}")

    with open(args.psg_cat, "rb") as f:
        psg_cat = pickle.load(f)
    print(f"PSG category embeddings loaded from: {args.psg_cat}")

    final_costs = np.zeros((1000,1000), dtype=np.float64)

    idx_1 = 0
    idx_2 = 0
    for query_graph, _ in tqdm(test_dataset, desc="Computing GED...", total=1000):
        for db_graph, _ in test_dataset:
            
            if idx_1 >= idx_2:
                idx_2 += 1
                continue

            sub_costs, query_costs, db_costs = compute_ged_costs(query_graph, db_graph, psg_cat)
            final_costs[idx_1, idx_2] = graph_edit_distance(dgl.from_networkx(query_graph),
                                                            dgl.from_networkx(db_graph),
                                                            node_substitution_cost=sub_costs,
                                                            G1_node_deletion_cost=query_costs,
                                                            G2_node_insertion_cost=db_costs)[0]
            idx_2 += 1
        idx_1 += 1
        idx_2 = 0
        print(f"{idx_1} Query Graphs Finished")
        
    with open(args.output, "wb") as f:
        pickle.dump( final_costs, f )
    print(f"GED computation finished and saved to: {args.output}")

if __name__ == "__main__":
    main()