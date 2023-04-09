# Thesis: Intubation prediction
Thesis project Manon Hendriks. Predicting intubation in ICU patients treated with HFNO therapy 

In this repository, two Python files are given that were used to predict intubations in ICU patients that received High Flow Nasal Oxygen therapy for at least 8 hours. The aggregated data model file contains a nested-cross validation (with 3 model types, 3 feature selection methods, and 3 scaling methods). The nested-CV is trained and tested with aggregated data. And in this code, an external validation data set is used to check the performance. The joint data model is likewise in content. However, it does not make use of aggregated data only but also data generated with LMEMs (Linear Mixed Effects Models).
The data used in the different files cannot be shared.
