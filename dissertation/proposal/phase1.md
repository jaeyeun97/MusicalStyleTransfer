Subject: Phase 1 - Yoon: Music Style Transfer using Generative Adversarial Network

Phase 1 Project Selection Status Report

Name: Charles Jae Yeun Yoon
College: King's
User Identifier: jyy24
Director of Studies: Timothy Griffin

Please complete 1, 2 and 3 below.

1. Please write 100 words on your current project ideas.

Current plan for the project is to mimic now popular image style transfer using generative adversarial network for music. I intend to train a network that takes a set of pairs of pieces of music, first the original song, and second a version of a song that would differ in quality, such as an "acoustic cover", and hope to be able to input a different set of musical snippets and make a synthetic cover of them. By mimicking `pix2pix` or other methods I would try to achieve such behavior, and it is also likely that I would preprocess the training and testing data. Facebook in a similar effort has produced results in their paper "A Universal Music Translation Network"[1]. 

2. Please list names of potential project supervisors.

I am supervised by Andrea Franceschini. The idea of the project was provided by Prof. Alan Blackwell.



3. Is there any chance that your project will involve any computing resources other than the Computing Service's MCS and software that is already installed there, for example: your own machine, machines in College, special peripherals, imported software packages, special hardware, network access, substantial extra disc space on the MCS.

If so indicate below what, and what it is needed for.

I will be using the Morganstanley, a machine in the Rainbow group for machine learning based project, for training. It has a Geforce Titan Xp for GPGPU Processing, and has Ubuntu 18.04 installed. I intend to install necessary software and tools on the system as I have gained the permissions to do so.

I will also be using my own desktop machine with a GTX 1080 and Arch Linux installed.

Some of my development will also be done on my laptop but more resource intensive work will be done mostly on the two machines above.

All code and work will be hosted on a private repository on Github.


[1]: arXiv:1805.07848 [cs.SD]
