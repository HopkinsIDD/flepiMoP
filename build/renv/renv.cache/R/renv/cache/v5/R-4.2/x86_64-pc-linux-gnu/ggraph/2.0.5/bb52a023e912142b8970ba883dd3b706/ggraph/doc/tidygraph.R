## ---- include=FALSE-----------------------------------------------------------
library(ggraph)
set_graph_style(family = 'Arial', size = 7, foreground = 'lightgrey', plot_margin = margin(0, 0, 0, 0))

## ---- message=FALSE-----------------------------------------------------------
library(tidygraph)

graph <- as_tbl_graph(
  data.frame(
    from = sample(5, 20, TRUE),
    to = sample(5, 20, TRUE),
    weight = runif(20)
  )
)
graph

## ---- eval=FALSE--------------------------------------------------------------
#  ggraph(graph, layout = 'fr', weights = "weight") +
#    geom_edge_link() +
#    geom_node_point()

## -----------------------------------------------------------------------------
ggraph(graph, layout = 'fr', weights = weight) + 
  geom_edge_link() + 
  geom_node_point()

## -----------------------------------------------------------------------------
ggraph(graph, layout = 'fr', weights = log(weight)) + 
  geom_edge_link() + 
  geom_node_point()

## -----------------------------------------------------------------------------
graph <- create_notable('zachary')

ggraph(graph, layout = 'fr') + 
  geom_edge_link() + 
  geom_node_point(aes(size = centrality_pagerank())) + 
  theme(legend.position = 'bottom')

## ---- message=FALSE-----------------------------------------------------------
ggraph(graph, 'matrix', sort.by = node_rank_leafsort()) + 
  geom_edge_point(aes(colour = centrality_edge_betweenness()), mirror = TRUE) + 
  theme(legend.position = 'bottom')

## -----------------------------------------------------------------------------
ggraph(graph, 'fr') + 
  geom_edge_link() + 
  geom_node_point() + 
  facet_nodes(~ group_infomap())

