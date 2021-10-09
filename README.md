# Gender-recognition

### Using the data from **CelebFaces Attributes (CelebA) Dataset** which consists of 202,599 images with 40 attributes, I trained an **Inception-v3** model with some custom composed output classes to recognize different gender in images.

## Inception-v3

![inception-v3](https://user-images.githubusercontent.com/27720480/136644979-7acad130-2bd9-4a28-a5bd-94026f4fd4e2.jpg)


## Training
### I used Pytorch for training my model as it is very low level and so we can cuztomize it better to our use case.
### As the data was very big, I only used 15,000 images and divided them into a train, test set of 10,000 and 15,000 images respectively.
### I used Adam as the optimizer and BCELoss for my loss function (as we were doing binary classification).

### I trained the model using a batch size of 200 and using 20 epochs with a learning rate of 0.001. I also tried learning rates of 0.01 and 0.0001 but they were either too big for the job or just too small. 

Data - https://www.kaggle.com/jessicali9530/celeba-dataset

Inception-v3 - https://paperswithcode.com/method/inception-v3

## Epoch
epoch:  0  loss:  31.90118408203125

epoch:  1  loss:  21.96202689409256

epoch:  2  loss:  19.68941020965576

epoch:  3  loss:  18.416278690099716

epoch:  4  loss:  18.11962977051735

epoch:  5  loss:  17.506986767053604

epoch:  6  loss:  17.26776686310768

epoch:  7  loss:  16.866083472967148

epoch:  8  loss:  16.82055914402008

epoch:  9  loss:  16.619915008544922

epoch:  10  loss:  16.39333549141884

epoch:  11  loss:  16.517942115664482

epoch:  12  loss:  16.329740971326828

epoch:  13  loss:  16.125786915421486

epoch:  14  loss:  15.839764282107353

epoch:  15  loss:  15.944547951221466

epoch:  16  loss:  16.04625627398491

epoch:  17  loss:  16.097503557801247

epoch:  18  loss:  16.122184738516808

epoch:  19  loss:  15.98683486878872


## Result
### After training the model we got an accuracy of 89.12% with a loss of 6.5 on our testing set as we correctly predicted 4456 times out of 5000.

## Extra
### Feel free to contact me if you have any question regarding my project.
