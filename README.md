# Using Machine Learning for Particle Tracking at the Large Hadron Collider
Source code of the project titled: _"Using Machine Learning for Particle Tracking at the Large Hadron Collider"_ of the ENLACE 2023 Summer Camp at UCSD.

This project was made for the ENLACE 2023 Research Summer Camp at the UCSD in a timeframe of 7 weeks and it's results were to be commited into a
poster (available in the repo) as part of the requirements for the University Students' projects. Most of the code is developed with the Pytorch module.

## Description
In the realm of particle physics, the Large Hadron Collider (LHC) stands as a colossal accelerator in Geneva, Switzerland, with its intricate network of superconducting magnets propelling particles to immense energies for experimental collisions. Within the LHC, the Compact Muon Solenoid (CMS) experiment captures the paths of charged particles through a powerful magnetic field, enabling the inference of charge and momentum through the reconstruction of particle paths via linear segments. In addressing the intricate task of discerning accurate tracks amidst complex particle interactions, we turn to Line Segment Tracking (LST). LST assembles tracks in a modular fashion, harnessing its inherent parallelizability to accommodate the demanding scenarios of the forthcoming High-Luminosity LHC (HL-LHC). The interplay between the convergence of the loss function during training – influenced by hidden layer size and model hyperparameters – and the potential for Machine Learning (ML) to amplify track reconstruction underscores the symbiotic relationship between ML models and the progressive reconstruction of LS to be considered as promising tools to be applied in the HL-LHC. This pivotal role of ML is exemplified in our architecture, which leverages Deep Neural Networks (DNNs) with varying hidden layer sizes to process Linear Segments (LS), culminating in an output neuron discerning the authenticity of the track. Our findings suggest that training a DNN can peform just as well as the simplest GNN for the task of classifying LS at the CMS.


## Results
lorem ipsum dolor
|             | **DNN > _X_** | **GNN > _Y_** | **Both** |
|-------------|---------------|---------------|----------|
| Real & Fake | 134876        | 128475        | 106670   |
| Real        | 49628         | 49628         | 48966    |
| Fake        | 85248         | 78847         | 57704    |

**Table 1.** Table of LS selected for a TPR > 0.95. _Note_:
**X** = 0.0328, **Y** = 0.0385.

lorem ipsum dolor
|             | **DNN > _X_** | **GNN > _Y_** | **Both** |
|-------------|---------------|---------------|----------|
| Real & Fake | 302497        | 313556        | 245207   |
| Real        | 51716         | 51717         | 51446    |
| Fake        | 250781        | 261839        | 193761   |

**Table 2.** Table of LS selected for a TPR > 0.99. _Note_:
**X** = 0.0032, **Y** = 0.0044.

## Credits
- Alejandro Dennis
- Abraham Flores
- Jonathan Guiang (mentor)
- Frank Wuerthwein (PI)
