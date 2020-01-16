# HEROKU_DEMO
Deployment of Car Model Classification using Flask and Heroku


### STEP 1: DATASET COLLECTION
Wrote a python script which uses a python package google_image_download to download the images from Google.
> My project mainly focus on two car models (i.e Maruti Suzuki Swift and BMW z4), thereby making it a binary image classification problem

> I collected nearly 250 images belonging to each class (50- test images and 200 -train images)

### STEP 2: TRAINING OF MODEL
Used Jupyter notebook for my training script and implemented MobileNet trained model from keras. After training the model, model is saved.

### STEP 3: DEPLOYMENT ON FLASK
Saved model was loaded and prediction was made on the flask app.

### STEP 4: HEROKU APP DEPLOYMENT
This was my first app deployment on Heroku and it is super easy to deploy your flask app on heroku which is a platform as a service.

## HAPPY PREDICTING YOUR CAR MODEL!
