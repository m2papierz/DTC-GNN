import dgl
import torch
import networkx as nx
import scipy.sparse as sp

from typing import List
from itertools import combinations
from qecsim.models.toric import ToricCode

from dtc_gnn.data_management.graph_transform_utils import GraphNode
from dtc_gnn.data_management.graph_transform_utils import GraphEdge


class GraphDataToTensorTransform:
    def __call__(
            self,
            g: dgl.DGLGraph
    ) -> dgl.DGLGraph:
        g.ndata["feat"] = g.ndata["feat"].to(torch.float32)
        g.edata["weight"] = g.edata["weight"].to(torch.float32)
        return g


class SyndromeToGraphTransform:
    def __init__(self, edges_constraint: int = 5):
        self._max_edges = edges_constraint

    def _syndrome_to_graph(
            self,
            stab_code: ToricCode,
            syndrome: List[int]
    ) -> None:
        """
        Creates graph representation of the syndrome.
        """
        self._graph_s = nx.Graph()

        # Add nodes to the graph based on syndrome
        stab_indexes = stab_code.syndrome_to_plaquette_indices(syndrome)
        for i, indices in enumerate(stab_indexes):
            node = GraphNode(indices=indices, code_dist=stab_code.n_k_d[-1])
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
            stab_code: ToricCode,
            syndrome: List[int]
    ) -> dgl.DGLGraph:
        self._syndrome_to_graph(
            stab_code=stab_code, syndrome=syndrome)
        dgl_graph = self._convert_nx_graph_to_dgl()
        return dgl_graph
