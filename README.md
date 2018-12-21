# Written by adc2181

# Phonaestesia

Generating images from phonemes


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
