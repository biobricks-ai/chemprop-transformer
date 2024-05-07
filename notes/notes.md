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

# april 23 model 1
100%|██████████████████████████████████████████████████| 669/669 [03:01<00:00,  3.69it/s]
epoch: 41       eval_loss: 0.7265       eval_acc: 0.9103        LR: 0.00000944151
![alt text](image-5.png)

# nprops = 5 model
>>> df[df['NUM_POS'] > 100].groupby('nprops').aggregate({'AUC': 'median', 'ACC': 'median', 'BAC': 'median', "cross_entropy_loss": 'median'})
             AUC
nprops          
0       0.630444
1       0.700298
2       0.728316
3       0.748020
4       0.761504

>>> evaltox21.groupby('nprops').aggregate({'AUC': 'median'})
             AUC
nprops          
0       0.651042
1       0.737103
2       0.778509
3       0.800564
4       0.829319


# mixture of experts model
This model is still very small and already outperforms all our property-transformer models.

evaldf.groupby(['nprops']).aggregate({'AUC': 'median','assay':'count'}).sort_values(by='AUC',ascending=False)
nprops                 
3       0.829222   4276
2       0.822098   4272
4       0.817691   4274
1       0.795135   3761
0       0.779107   2456