# icra2020
This is a source code for icra 2020 contributed paper: 
"**Real-time Continuous Hand Motion  Myoelectric Decoding by  Automated Data Labelling**"
 This code will be able to predict the hand motion of three degrees of freedom , 
 hand open/close, wrist flex/extension, wrist deviation, wrist rotation has been tested, feel free to train your model from scratch!

The annotation is being translated from Chinese to English, thank you for your patience

If it is helpful for your research,please cite our paper:

*Hu X, Zeng H, Chen D, et al. Real-time Continuous Hand Motion Myoelectric Decoding by Automated Data Labeling[J]. bioRxiv, 2019: 801985.*

ICRA citation is comming soon...
### System framework
Since all the work is done with python, I perfer to find a SDK of MYO running with the same language.
However,the compatibility I found in Windows seems not as good as in Linux. Therefore, considering a wearable application, 
I choose my Windows desktop to train the model, and use a raspberry pi as sub-proccessor to interface with MYO,
and to run the prediction model offline and real-time. 

Data training on a non-embedded device is time consuming,even the signal preprocessing.
If your computer is running with linux, you can run all the code in your computer as a indoor usage.


### Scripts Execution Pipeline
1. run **Collect_and_preprocess.py** first on your raspberry pi(tested on Zero and 3B+)
2. transfer the data to your Windows PC
3. run **HG_AEN.py** on your PC to train the data (much time efficiency)
4. transfer your trained matrices to your raspberry pi
5. run **unity_hand.py** on your raspberry pi



This project couldn't be accomplished without the effort of the open source commiunity：
* Thanks [dzhu](https://github.com/dzhu/myo-raw) for the share of Linux based SDK for MYO;
* Thanks [Alvipe](https://github.com/Alvipe/Open-Myo) for the share of raspbian based SDK for MYO;
* Thanks [Eli](http://billauer.co.il/peakdet.html) for the peak detection algorithm
* Thanks [wblgers](https://github.com/wblgers/tensorflow_stacked_denoising_autoencoder) for the Autoencoder framework
