# MitosisDomainAdaptation

Code for our BVM Paper entitled 
  "Learning New Tricks from Old Dogs - Inter-Species, Inter-Tissue Domain Adaptation for Mitotic Figure Assessment"


use as:

  python DomainAdaptation_ResNet.py <source_dataset> <target_dataset> <run>
  
  where <run> is a name of the run (to later on evaluate statistics over multiple runs)

Dependencies:

  - MulticoreTSNE (0.1)
  - fast.ai (tested with 1.0.52.dev0)
  - pytorch
  - matplotlib
  
