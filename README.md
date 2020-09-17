# Comparisson of autoencoder and PCA   

The code in this github repository shows dimension reduction performed by an autoencoder and PCA. he code heavily 
relies on the work of Urwa Muaz, see his jupyter notebook with the github link below. The reason this
repository exist was that we were unable to get his notebook to work.

https://github.com/muaz-urwa/PCA-vs-AutoEncoders

And read his article comparing autoencoders and PCA here:

https://medium.com/@urwamuaz/autoencoders-vs-pca-when-to-use-which-73de063f5d7

## Running the code with google colab
The easiest way to run this code is with [google colab](https://colab.research.google.com/notebooks/intro.ipynb#recent=true).
Make a new notebook, and make sure that ```Runtime -> change Runtime Type -> Hardware accelerator = GPU```,
and make the following code blocks:
```
!git clone https://github.com/2bben/PCAvsAutoencoder.git
```

```
%cd PCAvsAutoencoder/
!mkdir figures
!ls #just to check we are in the correct folder
```

In the last code block, we need to specify which example (by replacing ```0``` with ```1```) we want to run, as for  now 
there are only zeros, which non of the exmples are being ran. The:
1. value indicates _the linear case_
2. value indicates _the non linear case_
3. value indicates _3D linear case_
4. value indicates _3D curve case_

```
!python main.py 0 0 0 0
```

Every example will make 3 figures, one exact/true, one for PCA and one for Autoencoder. The figures can be found in:
```
PCAvsAutoencoder/figures/
```

