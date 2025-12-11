# YOLO Poker Hand Detection - Intro to Computer Vision Final Project  
### By: Angel Sanchez and Kayla Vincent

Poker hand classifier and card detection project using Ultralytics YOLO model with Kaggle card dataset. Cards are detected based on the symbols in their top left and bottom right corners. To classify cards as a specific hand, cards are first identified and then analyzed based on if they fit the requirements for any of the hands. The final model weights for the model we trained can be found in the runs/detect/train2/weights folder. Other evaluation metrics for the model are located in the train and val folders. 

Ultralytics GitHub: https://github.com/ultralytics/ultralytics  
Kaggle Card Dataset: https://www.kaggle.com/datasets/andy8744/playing-cards-object-detection-dataset

---

## Getting started - Setting up environment on VS Code

To run this project, we used the conda environment "cv" we created at the beginning of this course. This environment used Python version 3.11. 

To create this environment, perform the following steps:
1. Install Git:  
   The instructions can be found by [clicking here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)  
2. Clone the repository:  
   After navigating to the directory you want to store the local copy, run this line: 
   ``` 
   git clone https://github.com/angelgeck98/YOLO-PokerCards.git
   ```  
3. Install Miniconda:  
   Install using default settings https://docs.conda.io/en/latest/miniconda.html  
   Then, open **Anaconda prompt** in the Start menu.  
4. Create the environment:  
   ```
   conda create --name cv python=3.11
   ```

---

## Running the project
After opening up the project folder in VS Code, you will want to activate the cv environment:
```
conda activate cv
```
Ensure your prompt looks like this before proceeding:
```
(cv) C:\Users\johndoe>
```  

Then, install dependencies by running this line (this may take several minutes):
```
pip install -r requirements.txt
```  

To run the webcam Poker detector, run this line, which will open the webcam and begin searching for cards and poker hands! The file opens the front-facing camera, but the back-facing camera can be used by making slight changes to the webcam_poker_detector.py file. 
```
python webcam_poker_detector.py
```  
To quit, press 'q' on your keyboard, and the webcam detector will close. 

To run the file using the back-facing camera, change the camera_id int to "1" on line 14. Then, do the same thing by changing "self.camera_id" on line 106 to "1".
