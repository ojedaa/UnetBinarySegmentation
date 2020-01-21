# UnetBinarySegmentation
U-Net for binary segmentation of biomedical images, using the Weighted Binary Cross-Entropy as a loss function and the Dice-Loss as metric for saving the best model.  The WBCE loss function was designed with the purpose to mitigate the class imbalance, given more weight to the foreground class.
This code was used for Trypanosoma cruzi Parasite Segmentation. Receives an RGB color image with size of 256x256 and returns a-binary segmentation output of the same size.
