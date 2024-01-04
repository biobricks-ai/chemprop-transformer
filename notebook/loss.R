library(ggplot2)
library(readr)
library(dplyr)
library(scales) # For additional scale functions

# Read the data
data <- read_tsv('metrics/vaeloss.tsv') %>%
  mutate(iteration = row_number()) %>%
  select(epoch, iteration, loss, recloss, kloss) |>
  arrange(iteration) |>
  filter(iteration > 5) # skip first 5 high loss iterations

# Add a column indicating the start of a new epoch
data <- data %>%
  group_by(epoch) %>%
  mutate(start_of_epoch = (iteration == min(iteration))) %>%
  ungroup()

# Create the plot with a black theme and log scale for the y-axis
plot <- ggplot(data, aes(x = iteration)) +
  geom_line(aes(y = loss, color = "Total Loss"), size = 1) +
  geom_point(aes(y = loss, color = "Total Loss"), size = 2) +
  geom_line(aes(y = recloss, color = "Reconstruction Loss"), size = 1) +
  geom_point(aes(y = recloss, color = "Reconstruction Loss"), size = 2) +
  geom_line(aes(y = kloss, color = "KL Loss"), size = 1) +
  geom_point(aes(y = kloss, color = "KL Loss"), size = 2) +
  geom_vline(data = data %>% filter(start_of_epoch), aes(xintercept = iteration), color = "white", linetype = "dotted") +
  scale_color_manual(values = c("Total Loss" = "#1f77b4", "Reconstruction Loss" = "#d62728", "KL Loss" = "#2ca02c")) +
  scale_y_log10(labels = scales::comma) + # Log scale for y-axis
  labs(x = 'Iteration', y = 'Loss', title = 'VAE Losses Over Iterations', color = "Metric") +
  theme_minimal(base_size = 16) +
  theme(
    text = element_text(color = "white"),
    plot.background = element_rect(fill = "black"),
    panel.background = element_rect(fill = "black"),
    legend.background = element_rect(fill = "black", color = "black"),
    legend.position = "bottom",
    legend.title.align = 0.5,
    plot.title = element_text(color = "white"),
    axis.title = element_text(color = "white"),
    axis.text = element_text(color = "white"),
    axis.line = element_line(color = "white")
  ) +
  guides(color = guide_legend(title = "Metrics"))

# Create the directory and save the plot
dir.create('notebook/plots', recursive = TRUE, showWarnings = FALSE)
ggsave('notebook/plots/loss.svg', plot = plot, width = 12, height = 7, dpi = 300)
ggsave('notebook/plots/loss.png', plot = plot, width = 12, height = 7, dpi = 300)
