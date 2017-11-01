# STATE-PREDICTION-BY-ATTIRE-RECOGNITION
An image processing project

Every country has its own cultural costumes. Even in India every state follows different culture. States have there own festivals, language, songs, dance forms, food, believes and attire etc. 
One of the ways by which we can distinguish states from each other is from its traditional attire. From the current work we would like to predict the state/region from the image, depending upon what tradition dress person is wearing. 

Initially to train our model we have collected images from the following states as our dataset: 

• Gujarat (Ghagara Choli)

• Haryana (Daaman, Chunder and Kurti)

• Jammu and Kashmir (Burkha, Pheran and Scarf on head (called Taranga))

• Uttar Pradesh (Chikkan work)

• Punjab (Punjabi salwar suit)

# Features and States
Module 1 : collar detection : Haryana

Module 2 : upper body dress color detection: Lucknow (light, dull color, remaining states have bright color dresses)

Module 3 : silver jewellery covering major forehead : Kashmir

Module 4 : ratio of length of upper body dress to body height : punjab (patiala with short kurta) 

else: colorful Ghaghra choli : Gujarat

Digging deeper into dataset for states ,we found that each state had one unique feature that could help detect that state from others (One Vs rest) and thus in order to learn various techniques and add on our own logics we decided to work on these features. For predicting state of given input we decided to follow the steps shown in below flow chart. 
We first extract human from the given image using humanDetectionAlgorithm which uses harcascade and pass this as our image to work on to different modules. we did this so that we can find our required region of interest(ROI) for individual modules easily.We then apply our collar detection algorithm which is built over face detection algorithm.This will help us check whether the dress in image is a traditional haryana costume. If not then we pass our image through upper body dress color detector which tells whether the dress is of
dull color or bright color. This helps predict lucknow , as remaining states namely gujarat,punjab and kashmir have bright colored traditional costumes. If image is not classified still , we pass it through our next detector which is silver jwellery on forehead detector. This helps in uniquely identifying kashmiris as only their dress code involves forehead jewellery that extends on major part of
forehead.Haryanvi costume also includes forehead jewellery but since we already checked for haryana using collar detector , we need not worry about it.
The states which now remain are Punjab and Gujarat,Since both wear bright colorful dress therefore we classify punjabi traditional costume, which involves patiala and short kurta using dress length ratio to extract this feature. If this doesn't predict the state, we classify it as gujarat.
