import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import os
import inflect

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.client import device_lib 
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.models import save_model, load_model


car_df = pd.read_csv("./data/labeled_car_data.csv")

#=======================================================================================================================
# Using image_dataset_from_directory to load tensorflow datasets.
#=======================================================================================================================
def create_tensorflow_datasets(image_size, batch_size=32, shuffle = True, smart_resize = True, labels = 'inferred', label_mode = "int",
                               train_directory = "./data/organized/train/", val_directory = "./data/organized/val/",
                               test_directory = "./data/organized/test/", seed = 42):
    
    train_dataset = image_dataset_from_directory(directory = train_directory,
                                                 labels=labels,
                                                 label_mode = label_mode,
                                                 image_size=image_size,
                                                 batch_size=batch_size,
                                                 smart_resize=smart_resize,
                                                 seed=seed,
                                                 shuffle=shuffle)

    val_dataset = image_dataset_from_directory(directory = val_directory,
                                               labels=labels,
                                               label_mode = label_mode,
                                               image_size=image_size,
                                               batch_size=batch_size,
                                               smart_resize=smart_resize,
                                               seed=seed,
                                               shuffle=shuffle)

    test_dataset = image_dataset_from_directory(directory = test_directory,
                                                labels = labels,
                                                label_mode = label_mode,
                                                image_size=image_size,
                                                batch_size=batch_size,
                                                smart_resize=smart_resize,
                                                seed=seed,
                                                shuffle=shuffle)
    
    return train_dataset, val_dataset, test_dataset

#=======================================================================================================================
# This function takes as input a dataframe containing a models training history and adds a column to the dataframe
# that inidcates the number of epochs trained.
#
#=======================================================================================================================
def add_epoch_num_column(df, prior_epochs = 0):
    df['epoch_num'] = list(range(prior_epochs + 1, len(df) + 1 + prior_epochs))
    return df


#=======================================================================================================================
# This is a helper function to the plot_train_val_comparison function. T
# 
# This function takes as input a dataframe containing a models training history and returns both the 
# "most extreme value" (which can be either the minimum or maximum value depending on the setting for the direction parameter)
# as well as the epoch number where the most extreme value occured.
# 
# This information is used the plot_train_val_comparison to easily add symbols at the location of the maximum accuracy or 
# minimum loss when plotting these metrics vs the number of epochs trained.
#=======================================================================================================================
def get_extreme_value(df, metric='accuracy', direction = 'max'):
    
    metric_list = list(df[metric].to_numpy())
    
    if direction == 'max':
        metric_extreme_value = max(metric_list)
    elif direction == 'min':
        metric_extreme_value = min(metric_list)
        
    metric_extreme_index = metric_list.index(metric_extreme_value)
     
    epoch_list = list(df['epoch_num'].to_numpy())
    epoch_value = epoch_list[metric_extreme_index]
    
    return metric_extreme_value, epoch_value


#=======================================================================================================================
# This function is used to plot the model training metrics vs the number of epochs trained.
#=======================================================================================================================
def plot_train_val_comparison(df, metric = 'accuracy', epoch_threshold = None, train_color='hotpink', val_color='navy', learning_rate_trace=False, annotate_best_only = True,
                              max_val_indicator_symbol = True, max_val_indicator_text = True, min_val_indicator_symbol = True, min_val_indicator_text = True,
                              max_train_indicator_symbol = True, max_train_indicator_text=True, min_train_indicator_symbol = True, min_train_indicator_text = True, 
                              legend_font = 'xx-large', legend_frameon = False, val_max_marker_type = 'c*', val_max_marker_size = 12, train_max_marker_type = 'go',
                              train_max_marker_size = 8, val_min_marker_type = 'c*', val_min_marker_size = 12, train_min_marker_type = 'go', train_min_marker_size = 8):
    
    # Shortcut if we just want to annotate the best value.
    # Best is relative to the type of metric being plotted.
    # Best loss is min, but best accuracy is max.
    if annotate_best_only and metric == 'loss':
        min_val_indicator_symbol = True
        min_val_indicator_text = True
        min_train_indicator_symbol = True
        min_train_indicator_text = True
        max_val_indicator_symbol = False
        max_val_indicator_text = False
        max_train_indicator_symbol = False
        max_train_indicator_text = False
    elif annotate_best_only:
        max_val_indicator_symbol = True
        max_val_indicator_text = True
        max_train_indicator_symbol = True
        max_train_indicator_text = True
        min_val_indicator_symbol = False
        min_val_indicator_text = False
        min_train_indicator_symbol = False
        min_train_indicator_text = False
    
    sns.set_style('darkgrid')
    
    if epoch_threshold is not None:
        df = df.loc[df['epoch_num'] <= epoch_threshold, :]
        
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(18, 8), squeeze=True)

    sns.lineplot(data=df, x='epoch_num', y=metric, legend='full', color = train_color, label=metric, ax=axs)
    sns.lineplot(data=df, x='epoch_num', y=f'val_{metric}', legend='full', color = val_color, label=f'val_{metric}', ax=axs)
    
    if learning_rate_trace:
        
        if metric == 'loss':
            location = 'upper center'
        else:
            location = 'lower right'
            
            
        embedded_axis = inset_axes(parent_axes = axs,
                                   width="30%", # Width is 30% of the parent graph
                                   height=2.,   # Height is two inches
                                   loc=location)       
        

        sns.lineplot(data=df, x='epoch_num', y='lr', legend='full', color = 'black', label='learning_rate', linestyle = '--', ax=embedded_axis)
    
    axs.set_title(f"Training and Validation {metric} vs number of epochs", fontsize=28, weight='bold')
    axs.set_xlabel("Number of epochs trained", fontsize=20, weight='bold')
    axs.set_ylabel(metric, fontsize=20, weight='bold')
    axs.tick_params(axis='both', labelsize=16)
    
    # Add a symbol to show the maximum value on the validation curve.
    if max_val_indicator_symbol:
        max_val, epoch_val = get_extreme_value(df, metric=f"val_{metric}", direction = 'max')
        axs.plot(epoch_val, max_val, val_max_marker_type, markersize = val_max_marker_size)
    
    # Add text that says what the maximum value on the validation curve is.
    if max_val_indicator_text:
        
        if metric == 'loss':
            y_offset = 0.02
            x_offset = -2.0
            ha = 'right'
        else:
            y_offset = 0.01
            x_offset = 0.0
            ha = 'center'
        
        max_val, epoch_val = get_extreme_value(df, metric=f"val_{metric}", direction = 'max')
        style = dict(size=10, color = val_color, ha = ha)
        axs.text(epoch_val + x_offset, max_val + y_offset, f"{round(max_val, 3)}", **style)

    # Add a symbol to show the maximum value on the training curve.
    if max_train_indicator_symbol: 
        max_train, epoch_train = get_extreme_value(df, metric=metric, direction = 'max')
        style = dict(size=10, color = val_color, ha = 'center')
        axs.plot(epoch_train, max_train, train_max_marker_type, markersize = train_max_marker_size)
    
    # Add text that says what the maximum value on the 
    if max_train_indicator_text:
        
        if metric == 'loss':
            y_offset = 0.02
            x_offset = -2.0
            ha = 'right'
        else:
            y_offset = 0.01
            x_offset = 0.0
            ha = 'center'
        
        max_train, epoch_train = get_extreme_value(df, metric=metric, direction = 'max')
        style = dict(size=10, color = train_color, ha = ha)
        axs.text(epoch_train + x_offset, max_train + y_offset, f"{round(max_train, 3)}", **style)
    
    if min_val_indicator_symbol:
        min_val, epoch_val = get_extreme_value(df, metric=f"val_{metric}", direction = 'min')
        axs.plot(epoch_val, min_val, val_min_marker_type, markersize = val_min_marker_size)
    
    if min_val_indicator_text:

        if metric == 'loss':
            y_offset = 0.075
            x_offset = 0
        else:
            y_offset = 0.00
            x_offset = -1
        
        min_val, epoch_val = get_extreme_value(df, metric=f"val_{metric}", direction = 'min')
        style = dict(size=10, color = val_color, ha = 'right')
        axs.text(epoch_val + x_offset, min_val + y_offset, f"{round(min_val, 2)}", **style)
    
    if min_train_indicator_symbol:
        min_train, epoch_train = get_extreme_value(df, metric=f"{metric}", direction = 'min')
        axs.plot(epoch_train, min_train, train_min_marker_type, markersize = train_min_marker_size)
    
    if min_train_indicator_text:
        
        if metric == 'loss':
            y_offset = 0.075
            x_offset = 0
        else:
            y_offset = 0.00
            x_offset = -1
        
        min_train, epoch_train = get_extreme_value(df, metric=metric, direction = 'min')
        style = dict(size=10, color = train_color, ha = 'right')
        axs.text(epoch_train + x_offset, min_train + y_offset, f"{round(min_train, 2)}", **style)
    
    axs.legend(fontsize = legend_font, frameon = legend_frameon)
    plt.show()
    

#=======================================================================================================================
# When the custom cyclic learning rate class is utilized, a separate history dictionary is created that logs the 
# learning rate activity. https://github.com/bckenstler/CLR
#
# This function is used to take the more fine grained learning rate information in this dictionary (learning rate as 
# it was varied by batch) and take samples that are an epochs worth of batches apart (for this project there was 329
# iterations per epoch using batches of size 32). 
#
# This zoomed out view of the learning rate oscilliations can then be easily plotted as a subplot on a chart that
# shows a accuracy vs epochs or loss vs epochs. 
#=======================================================================================================================
def cyc_lr_hist_iterations_to_epochs(lr_df, hist_df, iterations_per_epoch = 329):
    
    lr_values_on_epochs = []
    
    for index, row in lr_df.iterrows():
        
        if row['iterations'] % iterations_per_epoch == 0:
            
            lr_values_on_epochs.append(row['lr'])
            
    hist_df['lr'] = lr_values_on_epochs
    
    return hist_df

#=======================================================================================================================
#  When 
#
#=======================================================================================================================
def stitch_histories(dfs):
    
    trimmed_dfs = []
    
    for df in dfs:
        
        df = add_epoch_num_column(df)
        
        # The last model checkpoint will be whenever the val_loss was a minimum
        epoch_last_model_save = df.loc[df['val_loss'] == df['val_loss'].min(), 'epoch_num'].to_numpy()[0]
        
        df = df.loc[df['epoch_num'] <= epoch_last_model_save, :]
        
        df.drop(columns=['epoch_num'])
        
        trimmed_dfs.append(df)
        
    combined_df = pd.concat(trimmed_dfs)
    
    combined_df = add_epoch_num_column(combined_df)

    combined_df.reset_index(drop=True, inplace=True)
    
    return combined_df


#=======================================================================================================================
# INCORRECT PREDICTION HISTOGRAM PLOTTING FUNCTIONS BELOW!
#=======================================================================================================================

#=======================================================================================================================
#
#
#=======================================================================================================================
def build_prediction_frequency_df(pred_data, car_class_to_name_map):

    predictions = pred_data['Top_1']

    freq_dict = {}

    for pred in predictions:
        if pred not in freq_dict.keys():
            freq_dict[pred] = 1
        else:
            freq_dict[pred] += 1

    all_frequency_data = {}
    all_frequency_data['class'] = []
    all_frequency_data['car_name'] = []
    all_frequency_data['predicted_count'] = []
    
    for key, value in freq_dict.items():

        all_frequency_data['class'].append(key)
        all_frequency_data['car_name'].append(car_class_to_name_map[key])
        all_frequency_data['predicted_count'].append(value)

    df = pd.DataFrame(all_frequency_data)

    df['class_and_name'] = 'Class: ' + df['class'].astype(str) + "\n" + df['car_name']

    return df

#=======================================================================================================================
#
#
#=======================================================================================================================
def split_car_name_to_multiple_lines(tick_labels):

    all_tick_labels = [tick_label.get_text() for tick_label in tick_labels]

    new_tick_labels = []

    for tick_label in all_tick_labels: 

        split_label = tick_label.split()

        new_label = ""

        for index, word in enumerate(split_label):

            if index == 2 or index == 4:
                new_label = new_label + "\n" + word
            else:
                new_label = new_label + " " + word

        new_tick_labels.append(new_label)
    
    return new_tick_labels

#=======================================================================================================================
#
#
#=======================================================================================================================
def plot_incorrect_predictions(pred_df, dataset_type = 'test', direction='worst', rank_by='average', num_classes = 5, offset = 0, top_n = 5, car_df=car_df, figsize = (20, 6)):

    sns.set_style('darkgrid')

    car_class_to_name_map = build_car_class_to_name_dict(car_df)
    
    prediction_analysis = get_sorted_prediction_analysis_list(pred_df = pred_df,
                                                              direction = direction,
                                                              analysis_size = num_classes,
                                                              n = top_n,
                                                              rank_by = rank_by)
    
    prediction_analysis = prediction_analysis[offset]

    df = build_prediction_frequency_df(pred_data = prediction_analysis, car_class_to_name_map = car_class_to_name_map)

    color_list = ['#003f5c', '#f95d6a', '#a05195', '#ff7c43', '#a020f0', '#1e90ff', '#228b22', '#556b2f', '#191970',
                   '#8b0000', '#808000', '#008080', '#4682b4', '#9acd32', '#daa520', '#8fbc8f']

    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = figsize, squeeze = True)

    sns.barplot(x='class_and_name', y='predicted_count', data=df, palette=color_list, ax=axs)
    
    axs.set_xticklabels(split_car_name_to_multiple_lines(axs.xaxis.get_majorticklabels()), fontdict = {'fontsize' : 14, 'fontweight' : 'bold', 'horizontalalignment' : 'right', 'rotation' : 45})
    axs.tick_params(axis='x', labelrotation=45, labelsize = 14)
    axs.tick_params(axis='y', labelsize = 14)
    axs.set_xlabel("Predicted Classes", fontsize=20)
    axs.set_ylabel("Count of Predicted Classes", fontsize=20)

    rank = offset + 1
    if rank > 1:
        p = inflect.engine()
        rank_word = p.ordinal(rank)
        rank_word = rank_word + " "
    else:
        rank_word = ""

    axs.set_title(f"Class: {prediction_analysis['Class']}     Car Name: {car_class_to_name_map[prediction_analysis['Class']]}\n\
        {rank_word}{direction} {rank_by} correct prediction percentage in the {dataset_type} dataset", fontsize=24, weight='bold')

    # Annotate the prediction percentages over the appropriate bars.
    total_height = np.sum(df['predicted_count'])

    for p in axs.patches: 
            
        # Percentage is the ratio of the bar height over the height of all bars conbined.
        percentage = f"{round((100 * (p.get_height() / total_height)), 2)}%"
            
        # Annotate on the left edge of the bar
        x = p.get_x()
            
        # Annotate just above the top of the bar
        y = p.get_y() + p.get_height() + 0.04
            
        #Perform annotation
        axs.annotate(percentage, (x,y), fontsize=16, fontweight='bold')

    plt.show()
    #return df
    return

#=======================================================================================================================
# INCORRECT PREDICTION IMAGES GRID FUNCTIONS BELOW!
#=======================================================================================================================

#=======================================================================================================================
#
#
#=======================================================================================================================
def sort_prediction_analysis_list(prediction_analysis_list, analysis_size, rank_by, direction):

    # Classes with the highest prediction success average
    if rank_by == 'average' and direction == 'best':

        return sorted(prediction_analysis_list, key = lambda pred_dict : pred_dict['Average_Correct'], reverse = True)[:analysis_size]

    # Classes with highest number of correct predictions
    elif rank_by == 'count' and direction == 'best':

        return sorted(prediction_analysis_list, key = lambda pred_dict : pred_dict['Num_Correct'], reverse = True)[:analysis_size]

    # Classes with the lowest prediction success average
    elif rank_by == 'average' and direction == 'worst': 

        return sorted(prediction_analysis_list, key = lambda pred_dict : pred_dict['Average_Correct'])[:analysis_size]

    # Classes with highest number of incorrect predictions
    elif rank_by == 'count' and direction == 'worst':

        return sorted(prediction_analysis_list, key = lambda pred_dict : pred_dict['Num_Incorrect'], reverse = True)[:analysis_size]



#=======================================================================================================================
#
#
#=======================================================================================================================
def check_for_invalid_inputs(direction, rank_by):

    valid_rank_bys = ['average', 'count']
    
    if rank_by not in valid_rank_bys:
        print("/n===================================================")
        print("Invalid input for parameter rank_by!")
        print(f"Valid options are: {valid_rank_bys}")
        print("===================================================\n")
        return True

    valid_directions = ['best', 'worst']

    if direction not in valid_directions:
        print("/n===================================================")
        print("Invalid input for parameter direction!")
        print(f"Valid options are: {valid_directions}")
        print("===================================================\n")
        return True

    return False

#=======================================================================================================================
#
#
#=======================================================================================================================
def generate_prediction_analysis_dict(pred_df):

    # List of columns in pred_df
    columns = list(pred_df.columns)

    # List of all columns that are formatted as Top_num
    top_n_columns = [column for column in columns if ("Top_" in column) and ("Accurate" not in column)]

    #  List of unique classes that predictions were predicted on
    unique_labels = list(np.unique(pred_df['label'].to_numpy()))

    # Dictionary of dictionaries to track prediction success.
    prediction_analysis = {f"Class_{label}" : {"Class" : label, "Num_Correct" : 0, "Num_Incorrect" : 0,"All_Top_Probability" : []} for label in unique_labels}

    # Add additional keys to each sub dictionary.
    for key, class_dict in prediction_analysis.items(): 
        for column in top_n_columns:
            class_dict[column] = []

    return prediction_analysis


#=======================================================================================================================
#
#
#=======================================================================================================================
def get_sorted_prediction_analysis_list(pred_df, direction, analysis_size, n, rank_by = 'average'):

    if check_for_invalid_inputs(direction = direction, rank_by = rank_by):
        return -1

    prediction_analysis = generate_prediction_analysis_dict(pred_df)

    # iterate down the rows of the prediction dataframe    
    for row_num, row in pred_df.iterrows():

        # Update if the prediction was correct or not.
        if row['label'] == row['Top_1']:
            prediction_analysis[f"Class_{row['label']}"]['Num_Correct']+=1
        else:
            prediction_analysis[f"Class_{row['label']}"]['Num_Incorrect']+=1
        
        # Update the "top_n" predictions for this class.
        for num_top_n in range(1, n+1):
            prediction_analysis[f"Class_{row['label']}"][f"Top_{num_top_n}"].append(row[f"Top_{num_top_n}"])
            prediction_analysis[f"Class_{row['label']}"]["All_Top_Probability"].append(row[f"Top_{num_top_n}"])

    prediction_analysis_list = [pred_dict for pred_dict in prediction_analysis.values()]

    for key, class_dict in prediction_analysis.items(): 
            class_dict['Average_Correct'] = class_dict['Num_Correct']  / (class_dict['Num_Correct'] + class_dict['Num_Incorrect'])    


    sorted_prediction_analysis = sort_prediction_analysis_list(prediction_analysis_list = prediction_analysis_list,
                                                               analysis_size = analysis_size,
                                                               rank_by = rank_by,
                                                               direction = direction)

    return sorted_prediction_analysis

#====================================================================================================================================
# By default this function will find the class with the lowest average correct prediction percentage, and will
# print two rows of images, where incorrectly classified images from this worst class are placed next to examples of the classes
# that the model classified them as. 
#
# Offset = 0 selects the class with the absolute worst average accuracy. Setting offset to 1, 2, etc will select the class with the 
# second, third, etc worst average accuracy.
#
# num_images = number of misclassified images from this class to show.
#
# class_filter allows you to specify a particular column(s) to filter the prediction dataframe down to before selecting the worst class.
#====================================================================================================================================
def display_misclassified_photos(pred_df, class_filter = None, direction = 'worst', rank_by ='average', num_images = 1, offset=0, n=5,
                                 dataset_type='test', verbose=False):
    
    # Dictionary mapping car classes to car names.
    car_class_to_name_dict = build_car_class_to_name_dict()

    # If we are prefiltering the dataframe down to a specific list of classes.
    if class_filter is not None:
        pred_df = pred_df.loc[pred_df['label'].isin[class_filter], :]

    # Get the {offset} most extreme (likely worse) class for this dataset.
    pred_data = get_sorted_prediction_analysis_list(pred_df, direction = direction, analysis_size= min(offset + 1, 196), n = n, rank_by = rank_by)[offset]

    class_num = pred_data['Class']

    # Filter down to incorrectly classified images in this worst class
    pred_df = pred_df.loc[(pred_df['label'] == class_num) & (pred_df['Accurate'] == 0), :]

    sample_size = np.min([len(pred_df.index), num_images])

    images_df = pred_df.sample(n = sample_size, axis = 0)
    images_df.reset_index(inplace=True, drop=True)

    fig, axs = plt.subplots(nrows = sample_size, ncols = n, figsize = (25, n * sample_size * 1.2))

    for row, df_row in images_df.iterrows():

        image_path = df_row['filepath']

        class_num = df_row['label']
        
        # Need to add 1 to take care of the discrepancy where the data directories are
        # named with classes 1 - 196 but the model actually predicts classes 0 - 195.
        if len(str(class_num + 1)) == 1:
            class_num_str = "00" + str(class_num + 1)
        elif len(str(class_num + 1)) == 2:
            class_num_str = "0" + str(class_num + 1)
        else:
            class_num_str = str(class_num + 1)

        car_name = df_row['label_names']

        pil_image =  tf.keras.preprocessing.image.load_img(image_path, color_mode="rgb", target_size=(520, 520), interpolation="nearest")

        axs[row][0].imshow(pil_image)
        
        image_filename = image_path.split("\\")[-1]
        axs[row][0].set_title(f"Correct Label\nImage: {image_filename}\nClass: {class_num}\nCar: {car_name}", weight = 'bold')

        for col_num, pred_info in enumerate([('Predicted Class', 'Top_1'), ('Second Highest Probability Class', 'Top_2'),
                                            ('Third Highest Probability Class', 'Top_3'), ('Fourth Highest Probability Class', 'Top_4')], start=1):
            
            title_str, column_str = pred_info

            pred_class = df_row[column_str]

            if len(str(pred_class + 1)) == 1:
                class_num_str = "00" + str(pred_class + 1)
            elif len(str(pred_class + 1)) == 2:
                class_num_str = "0" + str(pred_class + 1)
            else:
                class_num_str = str(pred_class + 1)

            if verbose:
                print("/n================================================================")
                print(f"Column num: {col_num}")
                print(f"Title Str: {title_str}")
                print(f"column str: {column_str}")
                print(f"pred_class: {pred_class}")
                print(f"class_num_str: {class_num_str}")
            
            base_dir = f"./data/organized/{dataset_type}/class_{class_num_str}/"
            
            # List all the files in the directory associated with the class that the model predicted
            # (predicted as either its Top_1 - Top_4 choice).
            file_choices = os.listdir(base_dir)
            
            # The model predicts a class not a photo, but we can randomly select a photo from this class
            # to be a representative of what the model predicted.
            file_choice = np.random.choice(file_choices)
            
            # Get the path to the file that is a memeber of the class the model predicted.
            file_choice_path = os.path.join(base_dir, file_choice)
            
            if verbose:
                print(f"file_choice_path: {file_choice_path}")
                print("================================================================/n")

            pil_image =  tf.keras.preprocessing.image.load_img(file_choice_path, color_mode="rgb", target_size=(520, 520), interpolation="nearest")

            axs[row][col_num].imshow(pil_image)
            axs[row][col_num].set_title(f"{title_str}\nImage: {file_choice}\nClass: {pred_class}\nCar: {car_class_to_name_dict[pred_class]}", weight = 'bold')
    return images_df


#=======================================================================================================================
# GENERATE PREDICTION FILES SECTION BELOW!
#=======================================================================================================================


#=======================================================================================================================
#
#
#=======================================================================================================================
def build_car_class_to_name_dict(df = car_df):

    labels = list(df['target'].to_numpy())

    # Target columns is 1 to 196, our model predicts classes 0 to 195
    labels = [label - 1  for label in labels]

    car_names = list(df['car_names'].to_numpy())

    class_to_name_dict = {}

    for label, name in zip(labels, car_names):

        if label not in class_to_name_dict.keys():
            class_to_name_dict[label] = name

    return class_to_name_dict


#=======================================================================================================================
#
#
#=======================================================================================================================
def add_accuracy_columns(pred_dict, labels, n):

    pred_dict['Accurate'] = [1 if pred_dict['label'][index] == pred_dict['Top_1'][index] else 0 for index in range(len(pred_dict['label']))]
    pred_dict[f'Top_{n}_Accurate'] = []

    for index, correct_class in enumerate(pred_dict['label']):

        top_n_preds = [pred_dict[f"Top_{top_class_num + 1}"][index] for top_class_num in range(n)]

        if correct_class in top_n_preds:
            pred_dict[f'Top_{n}_Accurate'].append(1)
        else:
            pred_dict[f'Top_{n}_Accurate'].append(0)

    return pred_dict


#=======================================================================================================================
#
#
#=======================================================================================================================
def add_car_name_columns(pred_dict, car_dict, n):

    pred_dict['label_names'] = [car_dict[class_num] for class_num in pred_dict['label']]

    for top_n_num in range(1, n+1):
        pred_dict[f'Top_{top_n_num}_Car_Name'] = [car_dict[class_num] for class_num in pred_dict[f'Top_{top_n_num}']]

    return pred_dict


#=======================================================================================================================
#
#
#=======================================================================================================================
def build_prediction_dictionary(preds, labels, group_filepaths, n, car_dict):

    # Take the class with the highest predicted probability as the prediction

    # Verify running using .argsort() with n=1 is the same as using .argmax!!! 
    # pred_classes = [np.argmax(pred) for pred in preds]

    top_n_pred_classes = [list(pred.argsort()[-n:][::-1]) for pred in preds]

    pred_dict = {}
    for pred_classes in top_n_pred_classes:
        for index, pred_class in enumerate(pred_classes):

            key = f"Top_{index + 1}"

            if key not in pred_dict.keys():
                pred_dict[key] = [int(pred_class)]
            else:
                pred_dict[key].append(int(pred_class))
            
    pred_dict['label'] = [int(label) for label in labels]
    
    if group_filepaths is not None:
        pred_dict['filepath'] = group_filepaths

    pred_dict = add_accuracy_columns(pred_dict, labels, n)
    
    pred_dict = add_car_name_columns(pred_dict, car_dict, n)
    
    return pred_dict


#=======================================================================================================================
#
#
#=======================================================================================================================
def concat_all_pred_files(save_path, n, prediction_file_base_name):

    all_files = os.listdir(save_path)

    pred_files = [filename for filename in all_files if filename.startswith(prediction_file_base_name)]

    full_pred_file_paths = [os.path.join(save_path, pred_filename) for pred_filename in pred_files]

    dfs = [pd.read_csv(filepath) for filepath in full_pred_file_paths]

    big_df = pd.concat(dfs)

    return big_df


#=======================================================================================================================
#
#
#=======================================================================================================================
def verify_all_files_predicted(df, test_data_path, dataset_type):

    files = []

    for dir in os.listdir(test_data_path):
        class_files = os.listdir(os.path.join(test_data_path, dir))
        files.extend(class_files)

    num_predictions = len(df.index)

    print("\n==================================================================")
    print(f"Number of examples in {dataset_type} dataset: {len(files)}")
    print(f"Number of predictions in the dataframe: {num_predictions}")
    print("==================================================================\n")


#=======================================================================================================================
#
#
#=======================================================================================================================
def build_pred_df(model, group_features, group_labels, group_filepaths, n, save_path, model_name, save_count, car_dict):

    # Path to save the predictions to.
    save = f"{save_path}{model_name}_{save_count}.csv"

    # Make prediction using this group of images.
    preds = model.predict(group_features)

    pred_dict = build_prediction_dictionary(preds, group_labels, group_filepaths, n, car_dict)

    # Convert the predictions
    df = pd.DataFrame(pred_dict)

    # Save the predictions made on this group.
    df.to_csv(save, index = False)

    # Make sure python has removed these from memory before building up the next group. 
    del df
    del preds
    del pred_dict
    del group_features
    del group_labels

    return 


#=======================================================================================================================
#
#
#=======================================================================================================================
def print_accuracy_results(df, n, dataset_type, model_name):

    # Total number of predictions made
    num_samples = len(df.index)

    top_n_accuracy_df = df.loc[df[f"Top_{n}_Accurate"] == 1, :]

    num_top_n_correct = len(top_n_accuracy_df.index)

    standard_accurate_df =  df.loc[df["Accurate"] == 1, :]

    num_correct = len(standard_accurate_df.index)

    top_n_accuracy = num_top_n_correct / num_samples
    accuracy = num_correct / num_samples

    print(f"\n================================ Accuracy Results on {dataset_type} dataset for {model_name} ================================")
    print(f"Total number of samples: {num_samples}\n")
    print(f"Top {n} Accuracy: {top_n_accuracy}")
    print(f"Number of Top {n} Correct Predictions: {num_top_n_correct}\n")
    print(f"Standard Accuracy: {accuracy}")
    print(f"Total Correct Predictions: {num_correct}")
    print("=======================================================================================================================\n")

    return {"Dataset_Type" : dataset_type,
            "Num_Samples" : num_samples,
            "Correct_Predictions" : num_correct,
            "Accuracy" : accuracy,
            f"Top_{n}_Accuracy" : top_n_accuracy,
            f"Num_Top_{n}_Correct" : num_top_n_correct}


#=======================================================================================================================
#
#
#=======================================================================================================================
def generate_top_n_prediction_files(test_dataset, model_filepath, save_path, n = 5, model_name = "resnet101_480_clrd2", dataset_type='test', test_group_size = 5,
                                    verbose=False, test_data_path = None, batch_size=32, car_df=car_df):
    
    image_filepaths = test_dataset.file_paths
    
    # Dictionary mapping class labels to car names.
    car_dict = build_car_class_to_name_dict(car_df)

    # Unique base string for all prediction files
    base_model_name = f"top_{n}_predictions_WITH_FILENAMES_{dataset_type}_dataset_" + model_name

    # Load the model we want to get predictions for
    model = load_model(model_filepath)

    # Empty numpy arrays to hold features and labels
    group_features = np.array([]).reshape(None, )
    group_labels = np.array([])
    group_count = 1
    save_count = 1

    # Iterate over the dataset we arre testing the model on.
    for features, labels in test_dataset:
        
        # Convert the features and labels from tensors to numpy arrays.
        features_numpy = features.numpy()
        labels_numpy = labels.numpy()
        
        # We evaluate the model using groups of data, to avoid reading all of the data into memory at once.
        # Continue appending features and labels to these arrays until we have a full group to evaluate with.
        group_features = np.append(group_features, features_numpy).reshape(-1, 520, 520, 3)
        group_labels = np.append(group_labels, labels_numpy).ravel()

        if group_count % test_group_size == 0:
            
            # Get the filepaths that go with the features and labels we have accrued.
            group_filepaths = image_filepaths[:(batch_size * test_group_size)]
            
            # Trim the list of remaining images
            image_filepaths = image_filepaths[(batch_size * test_group_size):]
            
            build_pred_df(model = model, 
                          group_features = group_features,
                          group_labels = group_labels,
                          group_filepaths = group_filepaths,
                          save_path = save_path,
                          model_name = base_model_name,
                          n = n,
                          save_count = save_count,
                          car_dict = car_dict)

            # Increment save counter for the next round of predictions.
            save_count+=1

            group_features = np.array([])
            group_labels = np.array([])

        group_count+=1
    
    # If if the number of samples predictions are being made for is not a multiple of batch_size * test_group_size 
    # Then there will be some remainder predictions that need to be made when the loop above exits. Make one last
    # prediction file for these remainder predictions. 
    if len(group_features) != 0:
        build_pred_df(model = model, 
                      group_features = group_features,
                      group_labels = group_labels,
                      group_filepaths = image_filepaths,
                      save_path = save_path,
                      n = n,
                      model_name = base_model_name,
                      save_count = save_count,
                      car_dict = car_dict)
    
    # Concatenate all intermediate prediction files to a single dataframe.
    all_predictions_df = concat_all_pred_files(save_path=save_path,
                                               n=n,
                                               prediction_file_base_name = base_model_name)

    if verbose and test_data_path is not None:
        verify_all_files_predicted(df = all_predictions_df, test_data_path = test_data_path, dataset_type = dataset_type)

    # Get the accuracy and top_n acuracy for these predictions.
    accuracy_results = print_accuracy_results(all_predictions_df, n = n, dataset_type = dataset_type, model_name = model_name)

    # Save the dataframe containing all of the predictions.
    final_save_path = f"{save_path}{base_model_name}_FINAL_SAVE_ALL_PREDICTIONS.csv"
    all_predictions_df.to_csv(final_save_path, index = False)

    all_predictions_df['label'] = all_predictions_df['label'].astype('int64')

    return all_predictions_df, accuracy_results


#=======================================================================================================================
#
#
#=======================================================================================================================
def print_accuracies(train_pred_df, test_pred_df, val_pred_df):
    
    corret_preds_train_standard = train_pred_df.loc[train_pred_df['Accurate'] == 1]
    train_accuracy_standard = len(corret_preds_train_standard) / len(train_pred_df)
    
    correct_preds_val_standard = val_pred_df.loc[val_pred_df['Accurate'] == 1]
    val_accuracy_standard = len(correct_preds_val_standard) / len(val_pred_df)
    
    correct_preds_test_standard = test_pred_df.loc[test_pred_df['Accurate'] == 1, :]
    test_accuracy_standard = len(correct_preds_test_standard) / len(test_pred_df)
    
    corret_preds_train_top_5 = train_pred_df.loc[train_pred_df['Top_5_Accurate'] == 1]
    train_accuracy_top_5 = len(corret_preds_train_top_5) / len(train_pred_df)
    
    correct_preds_val_top_5 = val_pred_df.loc[val_pred_df['Top_5_Accurate'] == 1]
    val_accuracy_top_5 = len(correct_preds_val_top_5) / len(val_pred_df)
    
    correct_preds_test_top_5 = test_pred_df.loc[test_pred_df['Top_5_Accurate'] == 1, :]
    test_accuracy_top_5 = len(correct_preds_test_top_5) / len(test_pred_df)
    
    print(f"\n====================================== Accuracy Report ======================================")
    print(f"Training: {train_accuracy_standard}")
    print(f"Training Top 5: {train_accuracy_top_5}\n")
    print(f"Validation: {val_accuracy_standard}")
    print(f"Validation Top 5: {val_accuracy_top_5}\n")
    print(f"Test: {test_accuracy_standard}")
    print(f"Test Top 5: {test_accuracy_top_5}")
    print("==============================================================================================\n")