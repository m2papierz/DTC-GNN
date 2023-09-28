import dgl
import torch
import numpy as np
import networkx as nx
import scipy.sparse as sp

from typing import List, Tuple
from qecsim.models.toric import ToricCode
from itertools import combinations


class GraphNode:
    def __init__(self, node_id: int, indices: tuple, code_dist: int):
        self._node_id = node_id
        self._indices = indices
        self._code_dist = code_dist
        self._is_x_stab = indices[0] == 1

    @property
    def id(self):
        return self._node_id

    @property
    def position(self) -> Tuple[int, int]:
        if self._is_x_stab:
            return self._indices[2], self._indices[1]
        else:
            return self._indices[2] + 0.5, self._indices[1] - 0.5

    @property
    def features(self) -> Tuple[int, float, float]:
        return (
            1 if self._is_x_stab else 0,
            self.position[1] / self._code_dist,
            self.position[0] / self._code_dist
        )


class GraphEdge:
    def __init__(self, node_a: GraphNode, node_b: GraphNode):
        self._node_a = node_a
        self._node_b = node_b

    @property
    def weight(self) -> float:
        a_pos = self._node_a.position
        b_pos = self._node_b.position
        return np.power(1 / np.max(
            [np.abs(a_pos[0] - b_pos[0]) + np.abs(a_pos[1] - b_pos[1])]
        ), 2)

    def get_nodes_ids(self) -> Tuple[int, int]:
        return self._node_a.id, self._node_b.id


class SyndromeToGraphTransform:
    def __init__(self, code: ToricCode, edges_constraint: int = 5):
        self._code = code
        self._code_dist = code.n_k_d[-1]
        self._max_edges = edges_constraint

    @staticmethod
    def _convert_nx_graph_to_dgl(
            nx_graph: nx.Graph
    ) -> dgl.DGLGraph:
        """
        Converts Networkx graph instance into DGL graph instance.
        """
        # Get adjacency sparse matrix from nx.Graph to create DGL graph instance
        adj_m_sparse = sp.csr_matrix(
            nx.adjacency_matrix(nx_graph).todense())
        dgl_graph = dgl.from_scipy(sp_mat=adj_m_sparse, eweight_name='weight')

        # Add node and edge features
        dgl_graph.ndata['feat'] = torch.tensor(
            list(nx.get_node_attributes(nx_graph, name='feat').values()))
        reshaped_weight = dgl_graph.edata['weight'].clone().detach().view(-1, 1)
        dgl_graph.edata['weight'] = reshaped_weight
        dgl_graph = dgl.add_self_loop(dgl_graph)

        return dgl_graph

    def _syndrome_to_graph(
            self,
            syndrome: List[int]
    ) -> nx.Graph:
        """
        Creates graph representation of the syndrome.
        """
        graph_s = nx.Graph()

        # Add nodes to the graph based on syndrome
        graph_nodes = []
        stab_indexes = self._code.syndrome_to_plaquette_indices(syndrome)
        for i, indices in enumerate(stab_indexes):
            node = GraphNode(
                node_id=i, indices=indices, code_dist=self._code_dist)
            graph_s.add_node(
                i, pos=node.position, feat=node.features)
            graph_nodes.append(node)

        # Create graph edges
        graph_edges = []
        for i, j in list(combinations(range(len(graph_nodes)), r=2)):
            edge = GraphEdge(node_a=graph_nodes[i], node_b=graph_nodes[j])
            graph_edges.append(edge)

        # Add edges to the graph
        for edge in sorted(graph_edges, key=lambda x: x.weight):
            node_a, node_b = edge.get_nodes_ids()
            node_a_const = len(graph_s.edges(node_a)) <= self._max_edges
            node_b_const = len(graph_s.edges(node_b)) <= self._max_edges
            if node_a_const and node_b_const:
                graph_s.add_edge(node_a, node_b, weight=edge.weight)
                graph_s.add_edge(node_b, node_a, weight=edge.weight)

        return graph_s

    def __call__(
            self,
            syndrome: List[int]
    ) -> dgl.DGLGraph:
        graph = self._syndrome_to_graph(syndrome)
        return self._convert_nx_graph_to_dgl(graph)
