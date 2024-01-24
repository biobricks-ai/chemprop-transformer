# Molecular Biological Similarity with Conditional Variational Autoencoder

Modeling the molecular biological similarity with conditional variational autoencoder. The model is trained on the [ChEMBL](https://www.ebi.ac.uk/chembl/) dataset using [BioBricks](https://biobricks.ai). 

## Development
1. install docker and nvidia-docker2
2. test nvidia-docker:  
`docker run --rm --gpus all nvidia/cuda:11.7.1-devel-ubuntu20.04 nvidia-smi`
3. build dockerfile
4. run dockerfile:  
`docker run --gpus all -it insilica/chemsim`

## References

* [Conditional Variational Autoencoder for Molecular Biological Similarity](https://arxiv.org/abs/XXX.XXXXX)

## License
MIT


# run service
curl -X GET "https://api.insilica.co/service/run/chemsim/predict?inchi=InChI=1S/C9H8O4/c1-6(10)13-8-5-3-2-4-7(8)9(11)12/h2-5H,1H3,(H,11,12)"
