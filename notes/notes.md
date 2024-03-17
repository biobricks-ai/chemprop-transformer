# Notes

## output models
/brick/working_mtt - this is a model that gets 80%+ AUC when evaluated with causal masking, but otherwise all known property-values. We wanted to improve on it by training more with a larger amount of masking. 

We build a property-sequence-transformer where propert-values are encoded in a sequence:

P1 V1 P2 V2 P3 V3 P4 V4 P5 V5 ...

The transformer mask is a usual causal mask but with a slight change:

1. When predicting for a property-name it is able to see the property-name.
2. When predicting a property-value it is a usual causal mask.  

This is done because the order of the property values is arbitrary. There are some nice properties of this approach:

1. The model can be trained on a single property-value pair.
2. The model can learn relationships between different properties.

Initially we tried including the selfies sequence and the property-value sequence in the output, 
with this approach we were able to get very low loss (0.001) but the median AUC across properties with a large number of positives and negatives remained below 70%.
The AUC across all properties combined was actually pretty high, I think this is because the properties that are strongly biased end up having easy predictions and
maybe there are many more of those than others? It is still strange that the AUC was biased lower though. 

![Alt text](image.png)

When using a very large multitask transformer on a selfies -> property-sequence we get:

![Alt text](image-1.png)

A much smaller transformer on the same task starts with higher loss and almost immediately converges to a slightly higher loss of 0.259

![Alt text](image-2.png)

# selfies input - 10 property-value output

After messing around a bit more we are now just using selfies as input and property-sequence as output. I tried a few different model sizes. The thing I found made the biggest difference was just training for longer. 

| Model | Description | AUC | ACC | BAC | LOSS | MEAN_EVAL_LOSS | BATCH |
|-------|-------------|-----|-----|-----|------|----------------|-------|
| 1.0 | very large model | 0.7283 | 0.6440 | 0.6447 | 0.25 | | |
| 1.1 | much smaller model & sum loss & L2 | 0.6914 | 0.6185 | 0.6178 | 921.94 | | |
| 1.2 | much larger model & sum loss & 15000 iter | 0.6653 | 0.6005 | 0.6029 | | 1.76361 | 128 |
| 1.2.1 | much smaller model & sum loss & 15000 iter | 0.7523 | 0.6707 | 0.6737 | | 1.5625 | 2048 |


I believe what is happening with the fast 'convergence' is that the model takes a bit of time to learn to predict trivial things like property-name from property-name, the end token, and padding tokens. After that the loss drops much more slowly. 

After getting an AUC over 0.8 we should try moving to a bigger machine, increasing the model size, and training for longer. 

# selfies input - single property-value output
It seemed worth trying a smaller set of property-values. So we tried just 1 property-value output sequence

| Model | Description | AUC | ACC | BAC | LOSS | MEAN_EVAL_LOSS | BATCH |
|-------|-------------|-----|-----|-----|------|----------------|-------|
| 1.3 | try a 1 property-value output sequence | 0.7523 | 0.6707 | 0.6737 | | 1.5625 | 2048 |

This did not seem to make a difference.

# let's try a decoder only with plain causal mask
perhaps the custom mask we wrote is not working and decoder only model is simpler and update this to take a selfies input and a selfies + propertyval output, then truncate the input severely.

| Model | Description | AUC | ACC | BAC | LOSS | MEAN_EVAL_LOSS | BATCH |
|-------|-------------|-----|-----|-----|------|----------------|-------|
| 1.5 | decoder only | 0.7581 | 0.6759 | 0.6759 | | 32.53 | 256 |
| 1.5.1 | decoder only big | 0.7704 | 0.68 | 0.68 | 31.4 | | 256 |
The model is quite small right now, and there is no sign of overfitting 

![Alt text](image-4.png)

So lets make it way bigger.

But, more importantly, we should get toxindex.com working with property categories, which is also finishing right now.

# a new approach
1. fixed prediction of properties with better mask
2. when back to encoder-decoder
3. predicting output of property-value sequence alone

| Model | Description                    | AUC  | LOSS | BATCH        |
|-------|--------------------------------|------|------|--------------|
| 1.5   | encoder-decoder pv-only output | 0.55 | 2.45 | 10000 x 128  |


>>> position_df
             AUC       ACC       BAC  count
nprops                                     
0       0.821531  0.755916  0.724965     10
1       0.522480  0.529046  0.508384     71

>>> position_df
             AUC       ACC       BAC  count
nprops                                     
1       0.803105  0.754282  0.699932    532
0       0.683921  0.643544  0.622061    508
2       0.663205  0.634772  0.617216    336