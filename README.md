# A Multi-Task Deep Learning Model for Estimating Earthquake Magnitude and Slip Distribution Using High-Rate GNSS, Strong-Motion Data, and Magnitude-Derived Priors
This repository contains the model implementation used in our study:
**“MPMSE: A Multi-Source Deep Learning Framework for Estimating Earthquake Magnitude and Slip Distribution”**  
The model integrates high-rate GNSS displacement and strong-motion velocity waveforms to jointly estimate:
- Earthquake magnitude (Mw)
- Maximum slip
- Slip area (rupture zone)
- Normalized slip distribution
- Final slip distribution reconstructed from maximum slip and normalized slip distribution

## Provided Files

- `models/model.py`: main model definition
- `models/model_parts.py`: model building blocks
- `models/data.py`: dataset and input loading utilities
- `models/trainer.py`: training and validation logic
- `models/losses.py`: loss functions
- `models/metrics.py`: evaluation metrics
- `models/config.py`: configuration definitions
- `train.py`: training entry point
- `config.example.json`: reference configuration file
