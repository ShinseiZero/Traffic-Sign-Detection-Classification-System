Traffic Sign Recognition System - User Manual
==========================================================================================================================================
Initial Setup
--------------------------------------------------------------------------------------------------------------
Before running the system, please ensure the following steps are completed:

1.OpenCV Installation:
  -Ensure you have OpenCV version 4.9.0 or higher installed on your system.
  -Make sure OpenCV is properly set up in your Visual Studio project. This includes linking the necessary OpenCV libraries and setting the include paths.

2.Input Folder:
  -Place all traffic sign images into the folder Inputs/Traffic signs/.
  -Ensure the images are in .png format, as the program specifically processes .png files.

3.Correct Image Format:
  -Verify that all images are in .png format. Any other format may cause the program to fail when reading the images.

Running the Program
---------------------------------------------------------------------------------------------------------------
When you run the program, the following sequence will occur:

1.Command Prompt Window:
 -A command prompt window will pop up for user interaction.
2.User Menu Options:
 -The prompt will display three main options for you to choose from:

  Option 1: Select this option to run segmentation on images in the Inputs/Traffic signs/ folder. The system will isolate the traffic
            signs based on their color (red, yellow, or blue) and save the segmented images in the Segmented/ folder.

  Option 2: Choose this option to extract color histogram features from the traffic sign images and train the SVM model. The model will be
            saved as svm_modelColor.yml and used to classify traffic signs based on color.

  Option 3: Use this option to extract HOG features (shape and texture) from the images and train the Random Forest classifier. The model
            will be saved as randomForest_modelHOG.yml and used to classify traffic signs based on shape.

  Option 0: Select this to exit the program.



Common Errors and Troubleshooting
----------------------------------------------------------------------------------------------------------------
Here are common errors that you might encounter, along with troubleshooting tips:

1. Error: "Cannot open image for reading."
   Solution: Ensure that the image files are located in the correct input directory (Inputs/Traffic signs/). Double-check the folder path
             and the existence of the .png files.

2. Error: "Failed to load the model."
   Solution: Ensure that the model files (svm_modelColor.yml or randomForest_modelHOG.yml) exist and are accessible in the directory. If
             the models do not exist, rerun the feature extraction and model training processes (options 2 or 3) to generate them.

3. Error: "No images found in the directory."
   Solution: Verify that the Inputs/Traffic signs/ folder contains images in .png format. Ensure that the folder path is correct and that
             the file extensions are properly set to .png.