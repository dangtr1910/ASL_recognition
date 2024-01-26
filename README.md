# ASL Recognition Integrated With Drone Technology

This script outlines the training and inference process for recognizing American Sign Language at letter level from A to Z. This work utilized two sets of stationary hand images for 26 letters, one for training and the other for unbiased inference. Using MediaPipe and several pre-processing steps that have been researched to optimize the accuracy and velocity, the scripts meticulously identified hands and the letters signed. Then, we used a Random Forest model to ensure an accurate but also seamless integration onto a Tello drone. We found that using a machine-learning based model outperforms a neural network-based model in both speed and accuracy. The research demonstrates efficacy with a light weight solution that offers real-time and accurate signed letter translation on a mobile device. It has the potential to progress into a personal translator for sign languages and can enhance communication experience for the deaf and the hard of hearing community. 

# Results
We recorded almost perfect accuracy when tested on the training dataset, which is equivalent to other similar research in the field, and achieved 62% accuracy when tested on a non-exposed dataset. Additionally, accuracy improved to 75% on webcam input and 68% on a moving drone camera input. 

# Demonstration 
Please find below a short video demonstration where the drone "reads" the hand movement and correctly outputs the letters. 
![trim_demo](https://github.com/dangtr1910/ASL_recognition/assets/108795992/8d724e80-fcee-4dd5-8936-b82a7b1b52e5)



