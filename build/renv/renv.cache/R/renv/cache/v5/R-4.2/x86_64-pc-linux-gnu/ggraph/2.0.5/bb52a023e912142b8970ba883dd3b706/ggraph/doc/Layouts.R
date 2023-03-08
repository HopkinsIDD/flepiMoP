## ---- message=FALSE-----------------------------------------------------------
library(ggraph)
library(tidygraph)

set_graph_style(plot_margin = margin(1,1,1,1))
graph <- as_tbl_graph(highschool)

# Not specifying the layout - defaults to "auto"
ggraph(graph) + 
  geom_edge_link(aes(colour = factor(year))) + 
  geom_node_point()

## -----------------------------------------------------------------------------
ggraph(graph, layout = 'kk') + 
  geom_edge_link(aes(colour = factor(year))) + 
  geom_node_point()

## -----------------------------------------------------------------------------
ggraph(graph, layout = 'kk', maxiter = 100) + 
  geom_edge_link(aes(colour = factor(year))) + 
  geom_node_point()

## -----------------------------------------------------------------------------
layout <- create_layout(graph, layout = 'eigen')
ggraph(layout) + 
  geom_edge_link(aes(colour = factor(year))) + 
  geom_node_point()

## -----------------------------------------------------------------------------
head(layout)

## -----------------------------------------------------------------------------
attributes(layout)

## -----------------------------------------------------------------------------
# An arc diagram
ggraph(graph, layout = 'linear') + 
  geom_edge_arc(aes(colour = factor(year)))

## -----------------------------------------------------------------------------
# A coord diagram
ggraph(graph, layout = 'linear', circular = TRUE) + 
  geom_edge_arc(aes(colour = factor(year))) + 
  coord_fixed()

## -----------------------------------------------------------------------------
graph <- tbl_graph(flare$vertices, flare$edges)
# An icicle plot
ggraph(graph, 'partition') + 
  geom_node_tile(aes(fill = depth), size = 0.25)

## -----------------------------------------------------------------------------
# A sunburst plot
ggraph(graph, 'partition', circular = TRUE) + 
  geom_node_arc_bar(aes(fill = depth), size = 0.25) + 
  coord_fixed()

## ---- fig.show='hold', results='hide'-----------------------------------------
graph <- as_tbl_graph(highschool) %>% 
  mutate(degree = centrality_degree())
lapply(c('stress', 'fr', 'lgl', 'graphopt'), function(layout) {
  ggraph(graph, layout = layout) + 
    geom_edge_link(aes(colour = factor(year)), show.legend = FALSE) +
    geom_node_point() + 
    labs(caption = paste0('Layout: ', layout))
})

## -----------------------------------------------------------------------------
graph <- graph %>% 
  mutate(friends = ifelse(
    centrality_degree(mode = 'in') < 5, 'few',
    ifelse(centrality_degree(mode = 'in') >= 15, 'many', 'medium')
  ))
ggraph(graph, 'hive', axis = friends, sort.by = degree) + 
  geom_edge_hive(aes(colour = factor(year))) + 
  geom_axis_hive(aes(colour = friends), size = 2, label = FALSE) + 
  coord_fixed()

## -----------------------------------------------------------------------------
ggraph(graph, 'focus', focus = node_is_center()) + 
  ggforce::geom_circle(aes(x0 = 0, y0 = 0, r = r), data.frame(r = 1:5), colour = 'grey') + 
  geom_edge_link() + 
  geom_node_point() + 
  coord_fixed()

## -----------------------------------------------------------------------------
graph <- tbl_graph(flare$vertices, flare$edges)
set.seed(1)
ggraph(graph, 'circlepack', weight = size) + 
  geom_node_circle(aes(fill = depth), size = 0.25, n = 50) + 
  coord_fixed()

## -----------------------------------------------------------------------------
set.seed(1)
ggraph(graph, 'circlepack', weight = size) + 
  geom_edge_link() + 
  geom_node_point(aes(colour = depth)) +
  coord_fixed()

## -----------------------------------------------------------------------------
ggraph(graph, 'treemap', weight = size) + 
  geom_node_tile(aes(fill = depth), size = 0.25)

## -----------------------------------------------------------------------------
ggraph(graph, 'treemap', weight = size) + 
  geom_edge_link() + 
  geom_node_point(aes(colour = depth))

## -----------------------------------------------------------------------------
ggraph(graph, 'tree') + 
  geom_edge_diagonal()

## -----------------------------------------------------------------------------
dendrogram <- hclust(dist(iris[, 1:4]))
ggraph(dendrogram, 'dendrogram', height = height) + 
  geom_edge_elbow()

## -----------------------------------------------------------------------------
ggraph(dendrogram, 'dendrogram', circular = TRUE) + 
  geom_edge_elbow() + 
  coord_fixed()

## -----------------------------------------------------------------------------
tree <- create_tree(100, 2, directed = FALSE) %>% 
  activate(edges) %>% 
  mutate(length = runif(n()))
ggraph(tree, 'unrooted', length = length) + 
  geom_edge_link()

## -----------------------------------------------------------------------------
graph <- create_notable('zachary')
ggraph(graph, 'matrix', sort.by = node_rank_leafsort()) + 
  geom_edge_point(mirror = TRUE) + 
  coord_fixed()

## -----------------------------------------------------------------------------
ggraph(graph, 'matrix', sort.by = node_rank_spectral()) + 
  geom_edge_point(mirror = TRUE) + 
  coord_fixed()

## -----------------------------------------------------------------------------
ggraph(graph, 'fabric', sort.by = node_rank_fabric()) + 
  geom_node_range(colour = 'grey') + 
  geom_edge_span(end_shape = 'square') + 
  coord_fixed()

## -----------------------------------------------------------------------------
ggraph(graph, 'fabric', sort.by = node_rank_fabric(), shadow.edges =TRUE) + 
  geom_node_range(colour = 'grey') + 
  geom_edge_span(aes(filter = shadow_edge), colour ='lightblue' , end_shape = 'square') + 
  geom_edge_span(aes(filter = !shadow_edge), end_shape = 'square') + 
  coord_fixed()

