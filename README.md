# Modernizing Generative Adversarial Text to Image Synthesis

This is (to my knowledge) the first publicly available PyTorch implementation of the [Generative Adversarial Text to Image Synthesis](https://arxiv.org/abs/1605.05396) paper that enables text-to-image generation for flowers. Training is based on the Oxford Flowers102 dataset and uses the GAN-CLS architecture from the paper listed above.

---

## Setting Up Training

This repository already contains pretrained models, finetuned text encoders, caption embeddings, and cache files that can be used for immediate generation. However, if you would like to change any hyperparameters or rerun training yourself from scratch, the following steps can be taken:

1. **Run `code/TextEncoderEvaluation.ipynb`**
    - This file contains code for evaluating different text encoders on their quality of embeddings on the Flowers102 dataset. You can replace any of the SentenceTransformer models as you see fit and experiment with the final accuracy values.
    - One thing to note – as discussed in the final report – is that the more advanced and finetuned models were found to perform worse than more basic models with lower reported accuracy. This was largely due to the generator becoming overly dependent on the text embeddings as opposed to the provided noise, which led to extremely deterministic outputs based on input text (regardless of what the base noise was).
    - Once this file is run, you can save any finetuned model or simply rely on an existing base SentenceTransformer model as you see fit.

2. **Run `code/DataLoader.ipynb`**
    - Once a text encoder is selected, the DataLoader file can be run to generate caption embeddings using your selected encoder.
    - This code will generate embeddings for the first 5 captions for each image in the dataset and save them to a file on disk (the pre-existing caption embeddings can be found in the `training encoding files` folder).
    - This code will also resize all images in the dataset to a specific size and save them to a file on disk as PyTorch tensors. This is done as an upfront cost in the data loading to massively boost speed during training — rather than directly opening image files for each iteration of the training loop, reading images as tensors from a file led to more than a 10x speedup.

3. **Run a training script**
    - `code/ClassicTraining.ipynb` contains a training structure that reimplements the original GAN-CLS training code (taken from [reedscot/icml2016](https://github.com/reedscot/icml2016/blob/db3f0c6d7c9ef3ef138cb15bf36592c7e19eb9a0/main_cls.lua)) as faithfully as possible.
    - `code/UpdatedTraining.ipynb` builds upon the classic structure by adding specific “GAN tricks” to the training that have been researched and found to improve stability and increase output quality. These include spectral normalization for the discriminator, adding noise to the real images used in training for a set number of epochs early on, and one-sided label smoothing.
    - `code/UpdatedPlusTraining.ipynb` builds upon the updated training by adding noise to text embeddings sent to the generator and adding an additional loss term to the generator that encourages diversity for different noise bases.
    - The logic behind creating these additional training scripts and an evaluation on each of them can be found in the final report.

4. **Choose a generator model**
    - The training scripts all save models and optimizers periodically. Once you pick a generator model, you’re ready to start generating images!

---

## Running Evaluation

1. **Run `code/Scoring.ipynb`**
    - This simple evaluation script will take the caption embeddings for the test data, generate images based on them, and then run the generated image and original written caption through CLIP.
    - The average CLIP score will be printed.
    - The best results from my training obtained an average CLIP score of around **0.2882**.

---

## Setting Up Generation

1. **Create captions**
    - Update `flower_captions.txt` to include descriptions of the flowers you want to generate images for.
    - Each line in the text file should contain its own isolated description of a flower.

2. **Run `code/Generation.ipynb`**
    - The code will first encode all of your provided captions using a text encoder of your choice. This should likely be the same text encoder that you used for the initial training.
    - The code will then take each embedded caption, generate X images for each caption, and save them to disk.

---

## Details

For details on motivation, implementation, evaluation, and more, please review the **Final Report** file.
