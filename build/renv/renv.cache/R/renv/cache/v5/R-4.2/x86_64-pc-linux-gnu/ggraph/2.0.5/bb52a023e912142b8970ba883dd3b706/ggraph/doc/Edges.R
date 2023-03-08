## ---- message=FALSE-----------------------------------------------------------
library(ggraph)
library(tidygraph)
library(purrr)
library(rlang)

set_graph_style(plot_margin = margin(1,1,1,1))
hierarchy <- as_tbl_graph(hclust(dist(iris[, 1:4]))) %>% 
  mutate(Class = map_bfs_back_chr(node_is_root(), .f = function(node, path, ...) {
    if (leaf[node]) {
      as.character(iris$Species[as.integer(label[node])])
    } else {
      species <- unique(unlist(path$result))
      if (length(species) == 1) {
        species
      } else {
        NA_character_
      }
    }
  }))

hairball <- as_tbl_graph(highschool) %>% 
  mutate(
    year_pop = map_local(mode = 'in', .f = function(neighborhood, ...) {
      neighborhood %E>% pull(year) %>% table() %>% sort(decreasing = TRUE)
    }),
    pop_devel = map_chr(year_pop, function(pop) {
      if (length(pop) == 0 || length(unique(pop)) == 1) return('unchanged')
      switch(names(pop)[which.max(pop)],
             '1957' = 'decreased',
             '1958' = 'increased')
    }),
    popularity = map_dbl(year_pop, ~ .[1]) %|% 0
  ) %>% 
  activate(edges) %>% 
  mutate(year = as.character(year))

## -----------------------------------------------------------------------------
ggraph(hairball, layout = 'stress') + 
  geom_edge_link(aes(colour = year))

## -----------------------------------------------------------------------------
ggraph(hairball, layout = 'stress') + 
  geom_edge_fan(aes(colour = year))

## -----------------------------------------------------------------------------
ggraph(hairball, layout = 'stress') + 
  geom_edge_parallel(aes(colour = year))

## -----------------------------------------------------------------------------
# let's make some of the student love themselves
loopy_hairball <- hairball %>% 
  bind_edges(tibble::tibble(from = 1:5, to = 1:5, year = rep('1957', 5)))
ggraph(loopy_hairball, layout = 'stress') + 
  geom_edge_link(aes(colour = year), alpha = 0.25) + 
  geom_edge_loop(aes(colour = year))

## -----------------------------------------------------------------------------
ggraph(hairball, layout = 'stress') + 
  geom_edge_density(aes(fill = year)) + 
  geom_edge_link(alpha = 0.25)

## -----------------------------------------------------------------------------
ggraph(hairball, layout = 'linear') + 
  geom_edge_arc(aes(colour = year))

## -----------------------------------------------------------------------------
ggraph(hairball, layout = 'linear', circular = TRUE) + 
  geom_edge_arc(aes(colour = year)) + 
  coord_fixed()

## -----------------------------------------------------------------------------
ggraph(hierarchy, layout = 'dendrogram', height = height) + 
  geom_edge_elbow()

## -----------------------------------------------------------------------------
ggraph(hierarchy, layout = 'dendrogram', height = height) + 
  geom_edge_diagonal()

## -----------------------------------------------------------------------------
ggraph(hierarchy, layout = 'dendrogram', height = height) + 
  geom_edge_bend()

## -----------------------------------------------------------------------------
ggraph(hairball, layout = 'hive', axis = pop_devel, sort.by = popularity) + 
  geom_edge_hive(aes(colour = year)) + 
  geom_axis_hive(label = FALSE) + 
  coord_fixed()

## -----------------------------------------------------------------------------
ggraph(hairball, layout = 'fabric', sort.by = node_rank_fabric()) + 
  geom_node_range(colour = 'grey') + 
  geom_edge_span(end_shape = 'circle') + 
  coord_fixed()

## -----------------------------------------------------------------------------
ggraph(hairball, layout = 'matrix', sort.by = bfs_rank()) + 
  geom_edge_point() + 
  coord_fixed()

## -----------------------------------------------------------------------------
ggraph(hairball, layout = 'matrix', sort.by = bfs_rank()) + 
  geom_edge_tile() + 
  coord_fixed()

## -----------------------------------------------------------------------------
ggraph(hairball, layout = 'linear') + 
  geom_edge_arc(aes(colour = year, alpha = stat(index))) + 
  scale_edge_alpha('Edge direction', guide = 'edge_direction')

## -----------------------------------------------------------------------------
ggraph(hierarchy, layout = 'dendrogram', height = height) + 
  geom_edge_elbow2(aes(colour = node.Class))

## -----------------------------------------------------------------------------
small_tree <- create_tree(5, 2)

ggraph(small_tree, 'dendrogram') + 
  geom_edge_elbow(strength = 0.75)

## -----------------------------------------------------------------------------
ggraph(small_tree, 'dendrogram') + 
  geom_edge_diagonal(strength = 0.5)

## -----------------------------------------------------------------------------
# Random names - I swear
simple <- create_notable('bull') %>% 
  mutate(name = c('Thomas', 'Bob', 'Hadley', 'Winston', 'Baptiste')) %>% 
  activate(edges) %>% 
  mutate(type = sample(c('friend', 'foe'), 5, TRUE))

## -----------------------------------------------------------------------------
ggraph(simple, layout = 'graphopt') + 
  geom_edge_link(arrow = arrow(length = unit(4, 'mm'))) + 
  geom_node_point(size = 5)

## -----------------------------------------------------------------------------
ggraph(simple, layout = 'graphopt') + 
  geom_edge_link(arrow = arrow(length = unit(4, 'mm')), 
                 end_cap = circle(3, 'mm')) + 
  geom_node_point(size = 5)

## -----------------------------------------------------------------------------
ggraph(simple, layout = 'linear', circular = TRUE) + 
  geom_edge_arc(arrow = arrow(length = unit(4, 'mm')), 
                start_cap = circle(3, 'mm'),
                end_cap = circle(3, 'mm')) + 
  geom_node_point(size = 5) + 
  coord_fixed()

## -----------------------------------------------------------------------------
ggraph(simple, layout = 'graphopt') + 
  geom_edge_link(aes(start_cap = label_rect(node1.name),
                     end_cap = label_rect(node2.name)), 
                 arrow = arrow(length = unit(4, 'mm'))) + 
  geom_node_text(aes(label = name))

## -----------------------------------------------------------------------------
ggraph(simple, layout = 'graphopt') + 
  geom_edge_link(aes(label = type), 
                 arrow = arrow(length = unit(4, 'mm')), 
                 end_cap = circle(3, 'mm')) + 
  geom_node_point(size = 5)

## -----------------------------------------------------------------------------
ggraph(simple, layout = 'graphopt') + 
  geom_edge_link(aes(label = type), 
                 angle_calc = 'along',
                 label_dodge = unit(2.5, 'mm'),
                 arrow = arrow(length = unit(4, 'mm')), 
                 end_cap = circle(3, 'mm')) + 
  geom_node_point(size = 5)

## -----------------------------------------------------------------------------
flaregraph <- tbl_graph(flare$vertices, flare$edges)
from <- match(flare$imports$from, flare$vertices$name)
to <- match(flare$imports$to, flare$vertices$name)
ggraph(flaregraph, layout = 'dendrogram', circular = TRUE) + 
  geom_conn_bundle(data = get_con(from = from, to = to), alpha = 0.1) + 
  coord_fixed()

