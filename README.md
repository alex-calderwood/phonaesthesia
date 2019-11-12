# Written by adc2181

# Phonaestesia

What would it mean to generate images from the sound of speech, not from the semantics of the words themselves (I built this with a generative model called a Variational Auto Encoder).

Project inspired by this incredible paper by Zach Leiberman: https://github.com/alex-calderwood/phonaesthesia/blob/master/papers/leiberman_paper.pdf

I never really got it to generate anything interesting, but I think that was a finding in itself, and I want to return to it one day.


# Run

Requires:

* spacy
* tensorflow
* opencv (for image processing)
* requests (for scraping ImageNet data)

1. To obtain images, run:

	python image_processing.py

Run until you have enough images. You will probably want to run this script in parrallel, or on multiple instances. 

2. To transform your data into 32 x 32 pixel images run:

	python resize.py

3. To generate new images based on the phonemes of the filenames in data/imagenet/scaled/ run:

	python vae_gan.py


Because other KALDI code is not a dependency for this project, I've only included the relavant directories from tedlium: lang/ and local/ to reduce the size of the upload. The idea is that this can integrate with KALDI/TEDLIUM by putting this code into the tedlium directory and running from there.
