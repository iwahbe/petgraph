use super::algo::{BoundedMeasure, FloatMeasure};
use super::visit::{EdgeRef, IntoEdges, IntoNodeIdentifiers, NodeCount, NodeIndexable};
use crate::graph::NodeIndex;
use crate::{algo::NegativeCycle, graph::node_index};
use std::collections::HashMap;
use std::hash::Hash;

/// Calculates the distance between any two nodes in the graph using the
/// \[Generic\] The Floyd-Warshall shortest paths algorithm. Calculates the
/// shortest path between each node in the graph in `O(V^3)` time, where `V` is
/// the number of nodes in the graph. Floyd-Warshall takes up `O(V^2)` space.
///
/// Details can be found on
/// [Wikipedia](https://en.wikipedia.org/wiki/Floyd–Warshall_algorithm).
///
/// Returns a PathCostMatrix that maps `(NodeId, NodeId)` to the cost of
/// traveling from the first node to the second.
///
/// # Example
/// ```rust
/// use petgraph::{Graph, algo::floyd_warshall};
///
/// let mut graph = Graph::new();
/// let a = graph.add_node(()); // node with no weight
/// let b = graph.add_node(());
/// let c = graph.add_node(());
/// let d = graph.add_node(());
/// let e = graph.add_node(());
/// let f = graph.add_node(());
/// let g = graph.add_node(());
/// let h = graph.add_node(());
/// // z will be in another connected component
/// let z = graph.add_node(());
///
/// graph.extend_with_edges(&[
///     (a, b, 3),
///     (b, c, 5),
///     (c, d, 2),
///     (d, a, 9),
///     (e, f, 1),
///     (b, e, -2),
///     (f, g, 8),
///     (g, h, -1),
///     (h, e, 0),
/// ]);
/// // a -----> b ----> e ------> f
/// // ^       |       ^       |
/// // |       v       |       v
/// // d <----- c       h <---- g
///
/// let costs = floyd_warshall(&graph).unwrap(); // handle negative cycle
/// assert_eq!(costs[(c,h)], 20);
/// assert_eq!(costs[(b,b)], 0);
/// assert_eq!(costs[(e,h)], 8);
///
/// let b_b_path: Vec<_> = costs.path(b, b).unwrap().into_iter().collect();
/// assert_eq!(b_b_path, vec![b]);
///
/// let b_a_path: Vec<_> = costs.path(b, a).unwrap().into_iter().collect();
/// assert_eq!(b_a_path, vec![b, c, d, a]);
///
/// // There is no valid path from e to b.
/// assert_eq!(costs[(e,b)], i32::MAX);
/// ```

pub fn floyd_warshall<G>(graph: G) -> Result<PathCostMatrix<G>, NegativeCycle>
where
    G: NodeCount + IntoNodeIdentifiers + IntoEdges + NodeIndexable,
    G::EdgeWeight: BoundedMeasure,
{
    let mut dist: PathCostMatrix<G> = PathCostMatrix::new(graph);
    graph
        .node_identifiers()
        .flat_map(|i| graph.edges(i))
        .for_each(|edge| {
            let (w, k) = dist.access_mut(edge.source(), edge.target());
            *w = *edge.weight();
            *k = Intermediary::Direct;
        });
    graph.node_identifiers().for_each(|vertex| {
        dist[(vertex, vertex)] = G::EdgeWeight::zero();
    });
    for k in graph.node_identifiers() {
        for i in graph.node_identifiers() {
            for j in graph.node_identifiers() {
                // Overflow guard.
                if dist[(i, k)] != G::EdgeWeight::infinite()
                    && dist[(k, j)] != G::EdgeWeight::infinite()
                    && dist[(i, j)] > dist[(i, k)] + dist[(k, j)]
                {
                    let ik = dist[(i, k)];
                    let kj = dist[(k, j)];
                    let (w, intermediary) = dist.access_mut(i, j);
                    *w = ik + kj;
                    *intermediary = Intermediary::Through(node_index(graph.to_index(k)));
                }
            }
        }
    }
    // A negative cycle was found.
    for k in graph.node_identifiers() {
        if dist[(k, k)] < G::EdgeWeight::zero() {
            return Err(NegativeCycle(()));
        }
    }

    Ok(dist)
}

/// Made to be used with with `floyd_warshall` algorithm. the cost of going from
/// each node to each other node.
///
/// ```rust
/// # use petgraph::{algo::floyd_warshall, Graph};
/// # use std::collections::HashMap;
/// # let mut get_graph = Graph::new();
/// # let node1 = get_graph.add_node(1);
/// # let node2 = get_graph.add_node(2);
/// # get_graph.extend_with_edges(&[(node1, node2, 1)]);
/// let g = get_graph;
/// let cost_matrix = floyd_warshall(&g).unwrap(); // Found negative cycle
///
/// assert_eq!(cost_matrix[(node1, node2)], 1);
///
/// // Or equivalently
///
/// let map: HashMap<_,_> = cost_matrix.into_hashmap();
/// assert_eq!(map.get(&(node1, node2)), Some(&1));
/// ```
pub struct PathCostMatrix<G>
where
    G: NodeIndexable + NodeCount + IntoNodeIdentifiers + IntoEdges,
    G::EdgeWeight: Clone,
{
    graph: G,
    weights: Vec<(G::EdgeWeight, Intermediary)>, /* Single vector improves cache
                                                  * locality and prevents
                                                  * unnecessary allocations. */
}

/// How two nodes form a path. Meant to annotate some edge (s,t).
#[derive(Clone, Copy)]
enum Intermediary {
    /// (s,t) form a path via some intermediary node index.
    Through(NodeIndex),
    /// (s,t) are directly connected
    Direct,
    /// There does not exist a path between (s,t)
    None,
}

impl<G> PathCostMatrix<G>
where
    G: NodeIndexable + NodeCount + IntoNodeIdentifiers + IntoEdges,
    G::EdgeWeight: Clone + BoundedMeasure,
{
    /// Generates `PathWeightMatrix` from a graph. The matrix is initialized,
    /// setting all weights to infinity.
    pub(crate) fn new(graph: G) -> Self {
        let weights =
            vec![(G::EdgeWeight::infinite(), Intermediary::None); graph.node_bound().pow(2)];
        PathCostMatrix { graph, weights }
    }

    /// Returns the path with lowest cost between `s` and `t` if it exists.
    pub fn path(&self, s: G::NodeId, t: G::NodeId) -> Option<impl IntoIterator<Item = G::NodeId>> {
        if s == t {
            Some(vec![s].into_iter())
        } else {
            let (_, k) = self.access(s, t);
            match k {
                Intermediary::Direct => Some(vec![s, t].into_iter()),
                Intermediary::Through(k) => {
                    let k = self.graph.from_index(k.index());
                    Some(
                        self.path(s, k)?
                            .into_iter()
                            .chain(self.path(k, t)?.into_iter().skip(1))
                            .collect::<Vec<_>>()
                            .into_iter(),
                    )
                }
                Intermediary::None => None,
            }
        }
    }

    fn access_mut(&mut self, s: G::NodeId, t: G::NodeId) -> &mut (G::EdgeWeight, Intermediary) {
        let index_0 = self.graph.to_index(s);
        let index_1 = self.graph.to_index(t);
        &mut self.weights[index_0 * self.graph.node_bound() + index_1]
    }

    fn access(&self, s: G::NodeId, t: G::NodeId) -> &(G::EdgeWeight, Intermediary) {
        let index_0 = self.graph.to_index(s);
        let index_1 = self.graph.to_index(t);
        &self.weights[index_0 * self.graph.node_bound() + index_1]
    }
}

impl<G> PathCostMatrix<G>
where
    G: NodeIndexable + NodeCount + IntoNodeIdentifiers + IntoEdges,
    G::EdgeWeight: Clone + BoundedMeasure,
    G::NodeId: Eq + Hash,
{
    /// Converts a `PathWeightMatrix` into a hashmap where the keys are index
    /// pairs and values are thier associated travel costs.
    pub fn into_hashmap(self) -> HashMap<(G::NodeId, G::NodeId), G::EdgeWeight> {
        self.graph
            .node_identifiers()
            .flat_map(|i1| self.graph.node_identifiers().map(move |i2| (i1, i2)))
            .map(|pair| (pair, self[pair]))
            .collect()
    }
}

impl<G> std::ops::Index<(G::NodeId, G::NodeId)> for PathCostMatrix<G>
where
    G: NodeIndexable + NodeCount + IntoNodeIdentifiers + IntoEdges,
    G::EdgeWeight: Clone + BoundedMeasure,
{
    type Output = G::EdgeWeight;
    fn index(&self, index: (G::NodeId, G::NodeId)) -> &Self::Output {
        let (w, _) = self.access(index.0, index.1);
        w
    }
}

impl<G> std::ops::IndexMut<(G::NodeId, G::NodeId)> for PathCostMatrix<G>
where
    G: NodeIndexable + NodeCount + IntoNodeIdentifiers + IntoEdges,
    G::EdgeWeight: Clone + BoundedMeasure,
{
    fn index_mut(&mut self, index: (G::NodeId, G::NodeId)) -> &mut Self::Output {
        let (w, _) = self.access_mut(index.0, index.1);
        w
    }
}

impl<G> Clone for PathCostMatrix<G>
where
    G: Clone + IntoNodeIdentifiers + NodeIndexable + NodeCount + IntoEdges,
    G::EdgeWeight: Clone,
{
    fn clone(&self) -> Self {
        Self {
            graph: self.graph,
            weights: self.weights.clone(),
        }
    }
}
