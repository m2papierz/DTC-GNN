import dgl
import torch
import networkx as nx
import scipy.sparse as sp

from typing import List
from itertools import combinations
from qecsim.models.rotatedtoric import RotatedToricCode

from dtc_gnn.data_management.transforms.graph_utils import GraphNode
from dtc_gnn.data_management.transforms.graph_utils import GraphEdge


class GraphSyndromeTransform:
    def __init__(self, edges_constraint: int = 5):
        self._max_edges = edges_constraint

    @staticmethod
    def _neighbours_counts(nodes_data):
        def calculate_distance(p_1, p_2):
            return max(abs(p_1[0] - p_2[0]), abs(p_1[1] - p_2[1]))

        n_counts_dict_x = {}
        n_counts_dict_z = {}
        for p1, p2 in combinations(nodes_data.keys(), 2):
            err1_is_x, err2_is_x = nodes_data[p1], nodes_data[p2]

            if calculate_distance(p1, p2) == 1:
                if err1_is_x:
                    n_counts_dict_x[p2] = n_counts_dict_x.get(p2, 0) + 1
                else:
                    n_counts_dict_z[p2] = n_counts_dict_z.get(p2, 0) + 1
                if err2_is_x:
                    n_counts_dict_x[p1] = n_counts_dict_x.get(p1, 0) + 1
                else:
                    n_counts_dict_z[p1] = n_counts_dict_z.get(p1, 0) + 1

        return n_counts_dict_x, n_counts_dict_z

    def _syndrome_to_graph(
            self,
            stab_code: RotatedToricCode,
            syndrome: List[int]
    ) -> None:
        """
        Creates graph representation of the syndrome.
        """
        self._graph_s = nx.Graph()

        # Add nodes to the graph based on syndrome
        nodes_data = {
            idx: stab_code.is_x_plaquette(idx)
            for idx in stab_code.syndrome_to_plaquette_indices(syndrome)
        }
        n_counts_dict_x, n_counts_dict_z = self._neighbours_counts(nodes_data)
        for i, (indices, is_x_plaquette) in enumerate(nodes_data.items()):
            node = GraphNode(
                indices=indices,
                code_dist=stab_code.n_k_d[-1],
                is_x=is_x_plaquette,
                n_counts_x=n_counts_dict_x.get(indices, 0) / len(nodes_data),
                n_counts_z=n_counts_dict_z.get(indices, 0) / len(nodes_data)
            )
            self._graph_s.add_node(i, pos=node.position, feat=node.features)

        # Create graph edges
        graph_edges = []
        nodes_pos = nx.get_node_attributes(self._graph_s, name='pos')
        for i, j in list(combinations(range(len(self._graph_s)), r=2)):
            edge = GraphEdge(ids=(i, j), node_a=nodes_pos[i], node_b=nodes_pos[j])
            graph_edges.append(edge)

        # Add edges to the graph
        for e in sorted(graph_edges, key=lambda x: x.weight):
            node_a_const = len(self._graph_s.edges(e.a_idx)) <= self._max_edges
            node_b_const = len(self._graph_s.edges(e.b_idx)) <= self._max_edges
            if node_a_const and node_b_const:
                self._graph_s.add_edge(e.a_idx, e.b_idx, weight=e.weight)
                self._graph_s.add_edge(e.b_idx, e.a_idx, weight=e.weight)

    def _convert_nx_graph_to_dgl(
            self
    ) -> dgl.DGLGraph:
        """
        Converts Networkx graph instance into DGL graph instance.
        """
        # Get adjacency sparse matrix from nx.Graph to create DGL graph instance
        adj_m_sparse = sp.csr_matrix(
            nx.adjacency_matrix(self._graph_s).todense())
        dgl_graph = dgl.from_scipy(sp_mat=adj_m_sparse, eweight_name='weight')

        # Add node and edge features
        dgl_graph.ndata['feat'] = torch.tensor(
            list(nx.get_node_attributes(self._graph_s, name='feat').values()),
            dtype=torch.float32)
        reshaped_weight = dgl_graph.edata['weight'].clone().detach().view(-1, 1)
        dgl_graph.edata['weight'] = reshaped_weight.to(torch.float32)
        dgl_graph = dgl.add_self_loop(dgl_graph)

        return dgl_graph

    def __call__(
            self,
            stab_code: RotatedToricCode,
            syndrome: List[int]
    ) -> dgl.DGLGraph:
        self._syndrome_to_graph(
            stab_code=stab_code, syndrome=syndrome)
        dgl_graph = self._convert_nx_graph_to_dgl()
        return dgl_graph
