# Evaluation of Explanation Correctness for Tabular Data
The code and documents contained in this repository supplement the PhD thesis "Evaluating Post Hoc Explanations Using Tabular Data: A Functionally-Grounded Approach" (publication forthcoming).

Some code used in this repository is sourced from <a href="https://github.com/irhete/predictive-monitoring-benchmark">the outcome-oriented predictive process monitoring benchmark</a> and <a href="https://github.com/renuka98/interpretable_predictive_processmodel/tree/master/BPIC_Data">a previous work on interpreting this benchmark</a>. We thank the authors of the benchmark on outcome oriented predictions, for providing the source code which enabled further study on the interpretability of the models:
* `BucketFactory.py`: Controls trace bucketing for event logs using scripts in the `bucketers` folder.
* `dataset_confs.py`: Preprocesses and identifies necessary features from event logs using scripts in the `preprocessing` folder.
* `DatasetManager.py`: Extracts case and inter-case features from event logs.
* `EncoderFactory.py`: Controls trace encoding for event logs using scripts in the `transformers` folder.
* `optimise_el_params.py`: Manages hyperparameter optimisation for PPA models.
* `prefix_training_and_testing_data.ipynb`: Prepares event log datasets for training.

Additionally, we have used the following scripts:
 * `acv_el_surrogate.py`: Generate an ACV surrogate model for a PPA predictive model.
 * `acv_tab_surrogate.py`: Generate an ACV surrogate model for a predictive model trained on tabular data.
 * `acv_el_surrogate.py`: Generate an ACV surrogate model for a PPA predictive model.
 * `el_white_box_fidelity.py`: Evaluates "glass box" fidelity for models trained on event logs.
 * `el_perm_fidelity.py`: Evaluates the correctness of perturbation-based importance for models trained on event logs.
 * `generate_el_model.py`: Trains PPA models
 * `generate_tab_model.py`: Trains models on tabular data
 * `learning.py`: Source code for the LINDA-BN XAI technique
 * `tab_white_box_fidelity.py`: Evaluates "glass box" fidelity for models trained on tabular data.
 * `tab_perm_fidelity.py`: Evaluates the correctness of perturbation-based importance for models trained on tabular data.
