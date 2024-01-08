library(ggplot2)
library(readr)
library(dplyr)
library(scales) # For additional scale functions

# Read the first three columns of the metrics file
data <- read_tsv('metrics/multitask_loss.tsv', col_names = c('type', 'batch', 'loss'), skip =1) 
data <- data |> mutate(batch = row_number()) |> mutate()
data <- data |> filter(batch> 20000)


data |> filter(type == "eval") |> pull(loss)
data |> filter(type == "eval") |> pull(loss) |> min()
# data |> filter(type == "scheduler")

data <- data |> filter(type != "train")
# Create the plot with a black theme and log scale for the y-axis
plot <- ggplot(data, aes(x = batch, y=loss, col=type)) + 
  geom_point(aes(color = type),alpha=0.8, size=1) +
  geom_smooth(size=2, alpha=0.5, col="red") +
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
  guides(color = guide_legend(title = "Metrics")) + 
  facet_wrap(~type, ncol = 3, scales = "free_y") 

# Create the directory and save the plot
dir.create('notebook/plots', recursive = TRUE, showWarnings = FALSE)
ggsave('notebook/plots/loss.svg', plot = plot, width = 12, height = 7, dpi = 300)
ggsave('notebook/plots/loss.png', plot = plot, width = 12, height = 7, dpi = 300)
