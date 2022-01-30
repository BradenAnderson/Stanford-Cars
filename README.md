
# Image Classification - Stanford Cars Dataset


***
***


### Table of Contents


[Data Preparation](#data-preparation)

[Beating The Baseline](#beating-the-baseline)

[Classifier Tuning](#classifier-tuning)

[Transfer Learning](#transfer-learning) 

[Fine Tuning](#fine-tuning)

[Analysis of Model Performance](#analysis-of-model-performance) 


***
***


### Data Preparation


**Notebooks:**
00_explore_stanford_cars_dataset.ipynb

01_create_train_val_test_directories.ipynb


**Description:**
The Stanford Cars dataset was provided to us in a MATLAB file format, which is not an ideal structure for working with data in Python. To make the data easier to work with, the 00_explore_stanford_cars_dataset notebook was created to parse the MATLAB file and create a .csv file containing same information. This file was named labeled_car_data.csv and contains the information needed to cross reference each image file name with the associated target label, car name, and coordinates the cars bounding box. Additionally, the 00_explore_stanford_cars_dataset contains functions that perform some initial exploration of the dataset by plotting examples of each target class. Lastly, a function is provided to explore the relative sizes of each image in the dataset. Prior to modeling all images will need to be resized to some constant size. In order to make an informed decision on the best constant image size, the mean and median image size across the dataset is calculated. This information is used to choose an image size that limits the amount of resizing that needs to be done on each image.

The 01_create_train_val_test_directories notebook continues to lay the foundation for efficient model building by distributing the images into folders that are structured in a manner that is compatible with the keras image_dataset_from_directory function. To this end, three main folders are created, one each for the training, validation and test datasets. The code then creates 196 subfolders inside each of the three top level dataset directories. The subfolders are labeled using the convention class_001 to class_196, because each subfolder will be filled with examples from one of the 196 target classes. 

I mention the details of the naming convention here because it is a potential pitfall that anyone implementing a similar process in their own projects will want to avoid. The keras image_dataset_from_directory function will infer class labels based on the alphanumeric ordering of the directory structure. This means a seemingly harmless change such as using the format class_1, ..., class_9, class_10 , ... , class_100, ... etc. will result in the target labels being scrambled in an undesireable way. Specifically, this ordering would result in class_1, class_10, class_100, class_101, class_102 being the first five directories found, and as a result these classes will be given target labels 0 - 4 by image_dataset_from_directory. To verify that your directory structures are set up such that the target labels are assigned to them in the order you want them assigned it is a good idea to list the folders in the directory using the os.walk() function. The order returned by os.walk() will be the order in which image_dataset_from_directory assigns the target labels. For those of you who are familiar with the O'Reilly machine learning book style, this is the type of thing that would be written with a scorpion next to it! I hope this note helps others from getting stung by incorrectly named directories! 

 After setting up the directory structure, the images are shuffled and divided according to a 65/20/15 training, validation and test split where the proportions of each target class across the three datasets is consistent with the proprotions in the full dataset. 


***


### Beating The Baseline


**Notebooks:**

1. 02_simple_convolutional_neural_networks.ipynb



**Description:**

Due to the relatively large number of target classes (196 different types of cars), the baseline accuracy for classification models on this dataset is fairly low at 0.008403 (0.8403%). The purpose of the 02_simple_convolutional_neural_networks notebook is to build a relatively simple model that can display predictive power by beating the baseline, and provide a starting point for future models to improve upon. The network architecture implemented for this task consists of a stack of 4 convolutional and max pooling layer pairs followed by a dense layer with 256 hidden units and a final dense classifier with 196 hidden units using softmax activations. This architecture was inspired by Francois Chollets book Deep Learning With Python, chapter 8. As we will analyze further in the final modeling analysis notebook, this simple model was able to significantly exceed the baseline, achieving a test set accuracy of 18.4%. A drawback of this model is that it also exhibited significant overfitting with a training accuracy 57.12% (shown as 32% in the training history plots, which record the training accuracy using predictions made when the dropout layer is active). This amount of overfitting occured depsite using both data augmentation and pretty high dropout rate of 0.5 between the final two dense layers, and is a challenge that we will work to overcome with future models. It is worth noting that no additional overfitting occured to the validation set, as the validation accuracy was 18.11%, which is nearly identical to the accuracy on the completely unseen test set.

An additional note of caution/lessons learned: This is the only modeling notebook that was run on Google Colab. Through this process I learned that Google Colab is very inefficient when asked to consistently read in large files from Google Drive (i.e. when image_dataset_from_directory is being used). If your data is in a simple .csv that can be read entirely into RAM once at the start of a notebook, you likely would never notice any delay. However when training a neural network and reading in images as batches (which avoids the need to ever hold the entire datase in RAM all at once), you will find that the majority of the programs execution time is spent reading in files. These slow file reads are not only inconvenient from a program execution time stand point, it is also very wasteful in terms of effective utilization of valuable GPU resources. The solution to overcome the slow read times is to copy the dataset to same directory that Colab Notebook is actually being run in. This process is described in greater detail in the 02_simple_convolutional_neural_networks notebook and is also shown in a very helpful blog post written by Oliver Muller linked to in the references section.

**References**
1. Deep Learning With Python, Francois Chollet

2. [Making image_dataset_from_directory efficient on Google Colaboratory](https://medium.datadriveninvestor.com/speed-up-your-image-training-on-google-colab-dc95ea1491cf)

***


### Classifier Tuning


**Notebooks:**

03_save_pretrained_model_outputs_numpy.ipynbv

03_save_pretrained_model_outputs_tfr.ipynb

03_dense_classifier_tuning.ipynb

**Description:**

In reality, the effort to tune hyperparameters with the goal of finding a high performing architecture for the models output classifier occured in an iterative manner that was simultaneous with much of the work performed in the transfer learning notebooks (described in detail in the next section). Although it is not chronologically accurate, much of the unique goals and lessons learned from these tasks are very different, therefore the decision was made to summarize these activites in separate sections in interest of providing clear and complete explainations.

The goal of the classifier tuning notebook series was to find a computationally efficient method of determining a high performing architecture and set of hyperparameters to use when building the model "top" (i.e. output classifier) that would be connected to a pre-trained base model (e.g. Resnet101). This is a challenging task because training the network with the base model included is a very computationally expensive operation which prohibits rapidly exploring a hyperparamter search space using standard tuning methods.

As a way to overcome this challenge and facilitate quick hyperparameter explorations the 03_save_pretrained_model_outputs_numpy notebook was created to calculate the outputs of the Resnet model and save them to disk as a numpy array. The idea was that once the Resnet outputs have been calculated and saved, a hyperparameter tuning test bed could created to simulate the full network configuration and explore hyperparameter options much more rapidly without the overhead of calculating the base model outputs at each training iteration. The drawback to this approach is that once the base model has been removed from the network we can no longer utilize the keras data augmentation preprocessing layers (because the base model is no longer there to continually calculate outputs for the randomly augmented images). Losing the ability to use data augmentation both significantly increases the likelihood of overfitting, and limits the amount in which the test bed accurately models the full network setup. These concerns certainly introduced doubt that the hyperparameters found by the test bed setup would accurately represent the hyperparameter values that would ultimately be the most effective in the full network configuration, however the hope was that if the tuning tests remained short the impact of these differences could be limited.

The method of saving outputs as numpy arrays turned out to be memory inefficient and was ultimately not implemented in the tuning setup, but it is briefly discussed here for completeness. Even when saving the numpy arrays in a compressed format using `numpy.savez_compressed` the files could still be several GB in size. Further, the saved numpy approach did not easily lend itself to taking advantage of the tensorflow dataset functions which make efficient use of RAM by reading in data batch wise. 

To take advantage of the tensorflow dataset features, the 03_save_pretrained_model_outputs_tfr notebook was created. The functions preformed by this notebook are identical to the 03_save_pretrained_model_outputs_numpy notebook with the exception that the datasets are saved in the TFRecords format rather than as numpy arrays. The TFRecords format is preferred because it makes it very easy to read in the data directly to a tensorflow dataset and take advantage of all of the tf.data.Dataset functionalities. 

As a way of creating a test enviornment that better modeled the full network setup, another idea was briefly explored but has not yet been implemented. The idea is that if the dataset was passed to the model using a custom tf.keras.utils.Sequence that implemented an on_epoch_end method which swapped out the contents of the dataset for a new set of preprocessed resnet outputs, we could better mimic the scenario where a fresh set of augmented images gets pushed through the model at each epoch. I am currently unsure if the final implementation of such an idea would be computationally reasonable from the perspective of still providing a significant speed increase as compared to simply using the full pretrained model in the hyperparameter tuning function, however if anyone reading this happens to know the answer I would be very interested in hearing what it is :). This is an area that may be explored further sometime in the future.

After creating tensorflow record files with the resnet outs (in the 03_save_pretrained_model_outputs_tfr notebook), the 03_dense_classifier_tuning notebook was used to tryout various network architectures and hyperparameter settings for the output classifier. The rounds of tuning significantly favored architectures with very few (1 or 0) additional dense layers prior to the final output layer with 196 hidden units. This result may have been at least partially influenced by the fact that rounds of tuning were kept short in order to mitigate the impact of overfitting that results from not having data augmentation. 

Ultimately I was not left with a high level of confidence that the test setup sufficiently modeled the full network setup such that the hyperparmeters found during testing would be good estimates of what would be effective in the complete configuration. Aside from the dataset swap on epoch end method previously mentioned, another idea which could have helped fight overfitting would have been to artifically increase the dataset size by including multiple copies of each augmented image. Due to time constraints this idea has not yet been implemented but may be an interesting area to explore further in the future.

The result from the 03_dense_classifier_tuning that was carried forward into the transfer learning notebooks was the identification of two relatively simple network top architecture that showed good performance in the test bed setting, these architectures are as follows:

1. Resnet Output --> Global Average Pool --> Flatten --> Dense 256, relu --> Dropout(0.5) --> Dense(196, softmax)

2. Resnet Output --> Global Average Pool --> Flatten --> Dense 480, relu --> Dropout(0.5) --> Dense(196, softmax) 

**References**

1. [Tensorflow TFRecords](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)
2. [Tensorflow TFRecords Options](https://www.tensorflow.org/api_docs/python/tf/io/TFRecordOptions)
3. [Helpful blog post for getting started with TFRecords](https://towardsdatascience.com/a-practical-guide-to-tfrecords-584536bc786c)


***

### Transfer Learning


**Notebooks:**

04_resnet_transfer_learning.ipynb

04_resnet_transfer_learning_lrdecay.ipynb

04_resnet_transfer_learning_lrdecay_480.ipynb

04_resnet_transfer_learning_cyclic_decay.ipynb


**Description:**

The goal of the 04 series notebooks was to perform several larger scale experiements that explored the impact of various output classifier architecture and learning rate hyperparameters. In each experiment the Resnet101 basemodel had all layers frozen and the weights in the network top were trained until convergence. The table below showns a summary of the experiements that were performed. For a detailed analysis of the training, testing and validation scores associated with each approach please reference the final modeling analysis notebook.


|    | Notebook                                 |   First Dense Layer |   Dropout | Learning Rate                                                                                                |
|---:|:-----------------------------------------|--------------------:|----------:|:-------------------------------------------------------------------------------------------------------------|
|  0 | 04_resnet_transfer_learning              |                 256 |       0.5 | Constant, 0.0005                                                                                             |
|  1 | 04_resnet_transfer_learning_lrdecay      |                 256 |       0.5 | Divide by 1.3 every 10 epochs.<br> Minimum allowed 1e-6                                                          |
|  2 | 04_resnet_transfer_learning_lrdecay_480  |                 480 |       0.5 | Manual decay, brief significant increase once plateau reached.<br>(triggered early stop shortly after)          |
|  3 | 04_resnet_transfer_learning_cyclic_decay |                 480 |       0.5 | *Cylic'exp_range'<br>base_learning_rate=4e-6<br>max_lr=4e-4<br>full cycle = 10 epochs<br>decay constant = 0.99994|     
|  4 | 04_resnet_transfer_learning_cyclic_decay |                 480 |       0.6 | *Cylic'exp_range'<br>base_learning_rate=6e-6<br>max_lr=1.2e-3<br>full cycle = 10 epochs<br>decay constant = 0.99997|


**Note**: For learning rates with * in the above table, see references 1 and 2. 

**References**
1. [Cylical Learning Rates for Traning Neural Networks, Leslie N. Smith 2015](https://arxiv.org/abs/1506.01186)
2. [Cyclical Learning Rate Implementation](https://github.com/bckenstler/CLR)

***

### Fine Tuning


**Notebooks:**

05_Fine_Tuning.ipynb

05_Fine_Tuning_Model_2.ipynb

**Description:**

An overview of one transfer learning workflow for utilizing a pretrained neural network (often times trained on the imagenet dataset) is as follows:

1. Choose the pretrained base model and remove the networks "top" (i.e. the dense classifier).
2. Attach a new dense classifier to the base model with output layers suitable for your task.
3. Freeze all weights in the base model.
4. Train the randomly initialized weights in the new layers you added until convergence.
5. Unfreeze one or more layers in the base model and train those layers together with the output classifier, using a very low learning rate. 

The final step in the workflow above is what we call "fine tuning". It is at this stage where we finally update the base model weights so they become more specific to the new task you are using them for. Some important considerations when deciding how many layers to unfreeze are as follows:

1. The more layers you unfreeze, the more weights you will be trying to train. Depending on the size of your dataset, attempting to train to many weights can quickly lead to overfitting.

2. The earlier layers in the network learn more generic representations that are likely to remain useful when transferred from task to ask. The upper layers in the network learn more complex relationships based on the simple found in the earlier layers, and these more complex relationships are more likely to be specific to the previous task the network was used for. This means we get much more bang for our buck training the upper layers of the network. 

The Resnet101 architecture is divided into 5 "blocks", each block containing several convolutional layers (among others such as pooling, batch normalization, etc.). With the considerations outlined above in mind, I decided to unfreeze the last 5 convolutional layers, which corresponded to half of the convolutional layers in the in the final block (block 5). 

This process was performed twice, using models 2 and 4 shown in the table above. This fine tuning stage led to significant improvements in model performance, corresponding to an approximate 20% increase in validation accuracy for each model. For a full breakdown of the performance differences, please reference the 06_model_analysis notebook.


***


### Analysis of Model Performance


**Notebooks:**

06_model_analysis.ipynb

model_analysis_helpers.py

**Description:**

This notebook is the culumination of all the above expierimentation. In this notebook the relative performances each model are thoroughly explored. Additionally, functions are provided to analyze the types of errors being made for the two models that were fully trained through the fine tuning stage. These error analysis plots are very eye opening. They show that even on classifications that the model got incorrect, the classes selected by the models are generally very reasonable (often almost the exact same car as the correct class just perhaps a different year or model). Additionally, many times an incorrect prediction actually had the correct class as the second or third highest probability. A common metric for multiclass classification problems with a higher number of classes is the "Top 5 Accuracy". Using the top 5 accuracy metric, if the model selected the correct class with any of its top 5 probabilities then the classification is counted as correct. 

The accuracy metrics for models 1 and 4 (reference table in section 4) are as follows:

**Model 1**

![model_1_ACC](https://media.git.generalassemb.ly/user/36215/files/c65e1f00-ef3e-11eb-9910-8b8e8d1b0268)


**Model 4**

![model_4_ACC](https://media.git.generalassemb.ly/user/36215/files/f4436380-ef3e-11eb-9f0d-ccc2d1558b5b)



**Note:** In order to keep the focus in the 06_model_analysis notebook completely on the metrics for each model, all of the code used to generate plots and perform analysis tasks was moved to the model_analysis_helpers.py file.  

***
