# Using Machine Learning for Particle Tracking at the Large Hadron Collider
Source code of the project titled: _"Using Machine Learning for Particle Tracking at the Large Hadron Collider"_ of the ENLACE 2023 Summer Camp at UCSD.

This project was made for the ENLACE 2023 Research Summer Camp at the UCSD in a timeframe of 7 weeks and it's results were to be commited into a
poster (available in the repo) as part of the requirements for the University Students' projects. Most of the code is developed with the Pytorch module.

## Description
In the realm of particle physics, the Large Hadron Collider (LHC) stands as a colossal accelerator in Geneva, Switzerland, with its intricate network of superconducting magnets propelling particles to immense energies for experimental collisions. Within the LHC, the Compact Muon Solenoid (CMS) experiment captures the paths of charged particles through a powerful magnetic field, aiming to distinguish accurate tracks amidst the complex particle interactions. Addressing the challenge of efficient track identification, the Line Segment Tracking (LST) algorithm emerges as a solution, reconstructing particle trajectories through linear segments. However, due to its inherent sequential nature, LST has limitations in handling increasingly intricate scenarios, highlighting the imperative of harnessing the power of Machine Learning (ML) techniques. This pivotal role of ML is exemplified in our architecture, which leverages Deep Neural Networks (DNNs) with varying hidden layer sizes to process Linear Segments (LS), culminating in an output neuron discerning the authenticity of the track. The convergence of the loss function during training, influenced by the hidden layer size and model hyperparameters, underscores the symbiotic relationship between advanced ML and the progressive analysis of particle tracks.


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
- Dr. Frank Wuerthwein (PI)
- Jonathan Guiang (mentor)
- Alejandro Dennis
- Abraham Flores
