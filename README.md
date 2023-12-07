# Using Machine Learning for Particle Tracking at the Large Hadron Collider
Source code of the project titled: _"Using Machine Learning for Particle Tracking at the Large Hadron Collider"_ of the ENLACE 2023 Summer Camp at UCSD.

This project was made for the ENLACE 2023 Research Summer Camp at the UCSD in a timeframe of 7 weeks and it's results were to be commited into a
poster (available in the repo) as part of the requirements for the University Students' projects. Most of the code is developed with the Pytorch module.

## Description
In the realm of particle physics, the Large Hadron Collider (LHC) stands as a colossal accelerator in Geneva, Switzerland, with its intricate network of superconducting magnets propelling particles to immense energies for experimental collisions. Within the LHC, the Compact Muon Solenoid (CMS) experiment captures the paths of charged particles through a powerful magnetic field, aiming to distinguish accurate tracks amidst the complex particle interactions. Addressing the challenge of efficient track identification, the Line Segment Tracking (LST) algorithm emerges as a solution, reconstructing particle trajectories piece-by-piece, forming linear segments. Notably, LST's modular approach allows for parallelizability, a crucial attribute in tackling the intricate scenarios posed by the forthcoming High-Luminosity LHC (HL-LHC). While LST thrives in parallel processing, it faces limitations in handling increasingly complex scenarios sequentially, thereby highlighting the imperative of harnessing the power of Machine Learning (ML) techniques. This pivotal role of ML is exemplified in our architecture, which leverages Deep Neural Networks (DNNs) with varying hidden layer sizes to process Linear Segments (LS), culminating in an output neuron discerning the authenticity of the track. The convergence of the loss function during training, influenced by the hidden layer size and model hyperparameters, underscores the symbiotic relationship between advanced ML and the progressive analysis of particle tracks.


## Results
As per the poster data, the training aspect of the model consisted in training two types of DNNs:
1. _Small_ DNN: A DNN where its arquitechture consists of 2 hidden layers of 32 neurons each and was trained with a learning rate of 0.002, a batch size of 1000 data entries per batch and for 50 epochs.
2. _Big_ DNN: A DNN wherre its arquitecture consists of 2 hidden layers of 200 neurons each and was trained with a learning rate of 0.002, a batch size of 1000 data entries per batch and for 100 epochs.

The focus of our results will be on those obtained with the aforementioned _Big_ DNN. The loss curve plot of both the training and testing datasets indicates that the model is indeed learning patters in the training dataset that _are applicable_ to those in the testing dataset and any other dataset for that matter; because both curves are trending downwards. If both curves where to diverge in any point in the epochs, we say taht the model is overfitting; simply put, that the model learned too much to the point that it became quite specialized in detecting patters **of the training dataset only**.

The prediction scores histogram indicates us at a glance that the vast mayority of data entries of our testing dataset are indeed labeled as _Fake_ and that the model is predicting them as such, hence the distribution of _Fake_ LS near the origin. On a similar fashion, the rest of the LS that are labeled as _Real_ are distributed to the far right, indicating that the model indeed is predicting those _Real_ LS as _Real_.
Nevertheless, we can appreciate a little overlap of _Real_ LS on top of the _Fake_ LS of the far left of the plot, which means that the model _mispredicted_ certain LS that are labeled as _Real_, and classified them as _Fake_; this a given with models related to Binary Classification.

On a related matter to the histogram overlaps, a better way to understand the rate at which is expected that the model will make this mispredictions iw with the help of a ROC Curve, which at a glance is telling us if the model is peforming accurate estimations of the _Real_ LS (TPR ) in contrast to those _Fake_ LS that are mispredicted as _Real_ (FPR). For the context of the comparison of the GNN vs the DNN, this curve tell us exactly by how much the model's peformance is similar to one another. On the first ROC Curve, we can observe tht the peformance of the _Small_ DNN is actually worse in comparison to the GNN, but in the case of the _Big_ DNN versus the GNN, we observe that this DNN is doing almost the same work with a simpler arquitechture in contrast to the complex nature of the GNN, which for technical reasos such as the time of training and the time of developing the model pipeline, the _Big_ DNN is better for our sake and purposes of classifying LS.

Continuing with the same plot, we plotted two square dots to further reference the estimated coordinates of where we could find a TPR > 0.95, and a TPR > 0.99 respectively, which in turn, can tell us the actual threshold to use for the model to actually comply with these TPR ranges.
The followup tables contain the distilled numbers of the total of LS that both the DNN and GNN classified given their respective threshold that satisfy the previous TPR boundaries.

On both **Table 1** and **Table 2** (also included in the poster) we observe that we get a similar distribution (Predicted Scores Histogram) of data that was predicted as _Real_ and _Fake_, the interesing detail is that, using the same testing dataset for both inferences with the DNN and the GNN, these models
are using a _substantial amount of the same LS for their predictions._

|             | **DNN > _X_** | **GNN > _Y_** | **Both** |
|-------------|---------------|---------------|----------|
| Real & Fake | 134876        | 128475        | 106670   |
| Real        | 49628         | 49628         | 48966    |
| Fake        | 85248         | 78847         | 57704    |

**Table 1.** Table of LS selected for a TPR > 0.95. _Note_:
**X** = 0.0328, **Y** = 0.0385.

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
