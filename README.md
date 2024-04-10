# xs-tswh-ai-dvc
Control the version of datasets and models of tswh

### Data Version Control Structure:

- train_data
    - emsd2_tswh
        - acc_data
        - acc_data_evaluate
            - abnormal
            - normal
        - acc_data_off
        - acc_data_on

# Meeting minute of TSWH Anomaly Detection - 08/04/2024
### Discussion:
- The possibility and method of transfer the models of TSWH to other site (e.g. HKE).
- The methodology to save the evaluation and validation result after training models with different datasets
- Logic for defining the abnormal datasets - based on the fault case (bearing) provided by andrew.w
### Next Steps:
- Develop a git repository with DVC for controlling the dataset version of different site to synchronize the dataset between XS and Tecky.
- For the new training, bypass the filtering function and based on the existing fault case (bearing) to define the abnormal datasets.
- Record the findings of training and evaluation of each version of the model.
- Write monthly report to state the progress of model development.
- Model training experiment with HKE dataset: To find out the volume of the dataset needed for transfer TSWH models to other site - 1 week? 2 weeks? 1 month? 3 months?
- Health Score Prediction with RPM: Implement the RPM normalization
- Health Score Prediction Diffusion: Replace existing GAN model architecture with Diffusion.