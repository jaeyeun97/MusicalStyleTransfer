
---
header-includes: 
- \usepackage{tikz}
- \usepackage{tikzscale}
- \usepackage{svg}
- \tikzset{every picture/.style={line width=0.75pt}}
subparagraph: True
numbersections: true
---

# Introduction



## Project Goals

This project aims to design and train a neural network architecture that learns a specific musical style, in order to transform a given piece of sound sample. A number of prior designs have been examined in order to find the most effective architecture for audio processing using machine learning. Audio manipulation using neural network has only gained traction recently, due to the success of the WaveNet model, designed by Google DeepMind. This model showed it possible to encode different content (i.e. words) for different style of audio (i.e. voice). This project follows the effort of a number of papers that were followed afterwards which tried to replicate a similar behavior in musical content. Namely, NSynth and GANSynth by Google Magenta Project, Universal Music Translation Network by Facebook AI Research[@citationneeded] and TimbreTron by [@citationneeded] were reviewed and implemented for the project. Neither of the two provided a code base during the implementation phase of the project.

Since the evaluation of a musical style is based on human perception of the audio quality, a carefully arranged set of questions have been prepared in order to test the efficacy of the networks.
[writeup after]


# Preparation

The general aim of this project is to perform digital audio processing using Deep Learning methods. Although Neural Networks are introduced in *Part IB Artificial Intelligence* and *Part II Machine Learning and Bayesian Inferences*, many of the contents of the project are beyond the coverage of Tripos courses; therefore, in this section I aim to clarify the starting point and the previous works I relied on to implement the project.

## Starting Point

* Machine Learning and Deep Neural Networks: Other than the introduction provided by the Part IB course, I was only exposed to an introduction to neural networks by an online machine learning course by Andrew Ng. Due to a previous UROP project, however, I had experience with feature engineering; in order to strengthen foundational knowledge on theories and foundations of machine learning and neural networks, I consulted Deep Learning by Ian Goodfellow. [@goodfellow] I had experience with PyTorch from previous projects, so I chose it for alacrity of development. 

* Audio Processing: I had little experience with Digital Signal Processing; although most material that will be introduced below require little domain knowledge, I reviewed the course materials for the Part II Digital Signal Processing course.

* Miscellaneous Tooling: Jupyter Notebook/Lab and Git were used for training monitoring and version control.

## Neural Networks

\begin{figure}[h]
	\includegraphics[width=0.5\textwidth]{./figures/simple_neural_network.png}
\centering
\caption{Machine Learning Approaches for Cancer Detection - Scientific Figure on ResearchGate. Available from: https://www.researchgate.net/figure/Simple-Neural-Network-Model-Representation\_fig1\_324592713 [accessed 8 May, 2019]}
\end{figure}

### Convolutional Neural Networks

\begin{figure}[h]
	\includegraphics{./figures/conv.tikz}
	\centering
	\caption{Example of a 2D Convolution function, with (a) as input, (b) as the filter, and (c) as the output. For each output channel there exists a filter with different parameters (\texttt{num\_out\_channel} = \texttt{num\_filters}). Therefore we can consider all Convolutional layer as fully connected in the channel dimension (given we do not group filters).}
\end{figure}

### Residual Neural Networks


\begin{figure}[h]
	\includegraphics[width=0.5\textwidth]{./figures/resnet.png}
	\centering
	\caption{https://towardsdatascience.com/an-overview-of-resnet-and-its-variants-5281e2f56035}
\end{figure}

## Style Transfer using Neural Networks

Gatys et al. introduced a mechanism for image style transfer based on pre-trained image classification networks. It uses the 19-layer VGG network, which has been proven to rival human performance in object recognition benchmarks. For each pre-selected layer in the network, it computes the content and style loss function. Given the original image $\vec{p}$ and generated image $\vec{x}$, the content loss is defined as the mean squared error function:

\begin{equation}
	\mathcal{L}_{content}(\vec{p}, \vec{x}) = \frac{1}{2} \sum_{l \in L} w_l \sum_{i,j} (F^l_{i, j} - P^l_{i, j})^2
\end{equation}

where $F^l$ and $P^l$ are feature representations from layer $l$ generated from $\vec{x}$ and $\vec{p}$, $L$ the set of selected layers, and $w_l$ the weighting factor of each layer.

To calculate the style loss of the system, Gatys et al. defines the style representation as the correlation between the filter responses within a single layer. To do so, for each layer $l$ they calculate the gram matrix $G$, which is the inner product of the output feature representation, given by $G_{ij} = \sum_{k} F^l_{ik} F^l_{kj}$. The style loss, alike the content loss, is defined as the mean squared error of two gram matrices $A$ and $G$, generated from the original image and the generated image respectively:

\begin{equation}
	\mathcal{L}_{style}(\vec{p}, \vec{x}) =  \sum_{l \in L} \frac{w_l}{4 N_l^2 M_l^2} \sum_{i, j} (G^l_{ij} - A^l_{i, j})^2
\end{equation}

where $N_l$ is the number of filters in a layer, $M_l$ the size of the feature map of a layer, and $w_l$ the weighting factor.

Such methods of style transfer for audio has been investigated by Ulyanov and Vadim Lebedev [@ulyanov]. Since they lack a multi-class classifier for audio representations that perform as well as VGG19, they used a single convolutional layer to generate the representation needed. While some of their results may sound sound, their results lack theoretical reasoning, contrary to their image style transfer counterpart. Therefore, other mechanisms using Generative Adversarial Networks have been investigated.

## GAN Style Transfers

Isola et al. introduced CycleGAN and `pix2pix`, which performs image style transfer without any pre-trained network. By using generative adversarial networks (GANs), these models learns not only the mapping between the two domain of images but also the loss function and therefore allow a general-purpose solution to image style transfer, without designing a domain-specific loss function. [@pix2pix]

### Generative Adversarial Networks

Generative Adversarial Nets are neural network architectures that arrange two neural networks, a generator and a discriminator, as a two player minimax game. The relationship of the generator and the discriminator is usually compared to a criminal and police; the generator aims to generate outputs that fails the discriminator, while the discriminator tries to correctly label the real observation apart from generated data. 

To represent the generator $G$, we define $p_z(\mathbf{z})$ as the input prior distribution, and $G(z; \theta_G)$ as the generative function that represents a multilayer neural network given parameters $\theta_G$. Similarly we define the discriminator function $D(x; \theta_D)$ which outputs a single scalar value that represent the probability of $x$ being a sample from $p_{data}(x)$.

The discriminator is trained to maximize $\log D(x)$ for $x \sim p_{data}(x)$ and minimize $\log D(G(x))$ for $x \sim p_g(x)$, or, in other words, to maximize the probability of assigning the correct source distribution. The generator, on the other hand, is trained to maximize $\log D(G(x))$ for $x \sim p_g(x)$. Given these characteristics, we can define the value function $V(G, D)$ for the minimax game between $G$ and $D$:

\begin{equation}
	V(G, D) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log (1 - D(G(z)))] \label{eq:gan}
\end{equation}

where $G$ is trained to minimize $V$ and $D$ to maximize $V$.

@goodfellow_gan proved that the global optimum of the value function above is when $p_g = p_{data}$, given that the two multilayer neural networks $G$ and $D$ have enough capacity. In practice, however, different rate of training between $G$ and $D$ may cause unstable training. Early in the training, when $G$ fails to generate convincing samples, $D$ can reject samples with high confidence. This causes $\log (1 - D(G(z)))$ to saturate and not provide gradient large enough to train $G$. To mitigate this problem we can train to maximize $\log D(G(z))$ instead. On the other hand, if $G$ trains much faster than $D$, $G$ may collapse too much of $p_g$ to model $p_{data}$. To mitigate this issue, we can update weights for $D$ multiple times for each update of $G$ weights.

We can create a conditional adversarial network that models $p(\mathbf{x} \mid c)$, by adding the condition $c$ as an input to both $G$ and $D$, as we will see in `pix2pix`.

### `pix2pix`

\begin{figure}[h]
	\includegraphics[]{figures/pix2pix.tikz}
\centering
\caption{\texttt{pix2pix} model dataflow. Each $\mathcal{L}$ is a loss value calculated by computing the mean squared error between the discriminator output and the correct label. \label{fig:pix2pix}}
\end{figure}

`pix2pix` use conditional adversarial networks to create a general-purpose image style transfer mechanism, given a one-to-one paired dataset. Each pair of data observation has a vector sampled from a domain $X$ (i.e. a photograph), and a range $Y$ (i.e. a painting). We define a generator function $G(\vec(x), \theta_G)$ much alike to the Generative Adversarial Network shown above, but have it model a mapping from $X$ to $Y$. We also define the data distribution of $X$ as $p_{data}(X)$ and that of $Y$ as $p_{data}(Y)$. With the given definition, we modify the GAN value function defined in equation \ref{eq:gan} to:

\begin{multline}
	V(G, D) = \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{X}), \mathbf{y} \sim p_{data}(\mathbf{Y})}[\log D(x, y)] \\ +  \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{X}), \mathbf{z} \sim p_{z}(\mathbf{z})}[\log (1 - D(x, G(x, z)))] \label{eq:pix2pix}
\end{multline}

The implementation would look like figure \ref{fig:pix2pix}, where each $\mathcal{L}$ is a loss value calculated by computing the mean squared error between the discriminator output and the correct label.  To use `pix2pix` for music style transfer, however, it is necessary to have a set of paired data, which is hard to acquire. Therefore, I took a closer look into CycleGAN, which does not require a paired dataset.

### CycleGAN

\begin{figure}[h]
	\includegraphics[]{figures/cyclegan.tikz}
\centering
\caption{Dataflow of CycleGAN architecture. $G_{A \rightarrow B} = G$ and $G_{B \rightarrow A} = F$ from equations \ref{eq:cyclegan_1} and \ref{eq:cyclegan_2}. \label{fig:cyclegan}}
\end{figure}

CycleGAN has been introduced to make style transfer possible without a paired dataset. It does so by training a pair of generators and discriminators, arranged as shown on figure \ref{fig:cyclegan}. We define the following:

* two dataset domain $A$ and $B$, and their data distribution $p_{data}(\mathbf{A})$ and $p_{data}(\mathbf{B})$
* Prior distribution for the noise variable $z \in Z$, $p_z$
* Generator $G: A, Z \rightarrow B$ and its output distribution $p_G$ which models $p_{data}(\mathbf{B})$
* Generator $F: B, Z \rightarrow A$ and its output distribution $p_F$ which model $p_{data}(\mathbf{A})$
* Discriminator $D_{A}: A \rightarrow \mathbb{R}[0, 1]$, which estimates the probability of its input being a sample from $p_{data}(\mathbf{A})$
* Discriminator $D_{B}: B \rightarrow \mathbb{R}[0, 1]$, which estimates the probability of its input being a sample from $p_{data}(\mathbf{B})$

where $\mathbb{R}[0, 1] = \{x \mid x \in \mathbb{R} \land 0 \leq x \leq 1 \}$. Given above, we define two GAN Value functions as:

\begin{align}
	\begin{split}
		V_{A \rightarrow B}(G, D_B) &= \mathbb{E}_{y \sim p_{data}(\mathbf{B})}[ \log D_B(y)]  \\
		&\quad + \mathbb{E}_{x \sim p_{data}(\mathbf{A}), z \sim p_{z}(\mathbf{z})} [\log (1 - D_B(G(x, z)))] \label{eq:cyclegan_1}
	\end{split} \\
	\begin{split}
		V_{B \rightarrow A}(F, D_A) &= \mathbb{E}_{x \sim p_{data}(\mathbf{A})}[ \log D_A(x)]  \\
		&\quad + \mathbb{E}_{y \sim p_{data}(\mathbf{B}), w \sim p_{z}(\mathbf{w})} [\log (1 - D_A(F(y, w)))] \label{eq:cyclegan_2}
	\end{split} 
\end{align}

where the model optimizes for $\displaystyle \min_G \max_{D_B} V_{A \rightarrow B}(G, D_B)$ and $\displaystyle \min_F \max_{D_A} V_{B \rightarrow A}(F, D_A)$. Intuitively speaking, $G$ aims to generate an image that looks alike those in $B$ (e.g. a Van Gogh painting) from an input image of $A$ (e.g. a photograph), and vice versa for $F$.

\paragraph{Cycle Consistency} Alongside GAN loss, we define Cycle Consistency Loss which allows us to train an unpaired dataset. While the GAN loss can be enough to optimize $G$ and $F$, with networks of large enough capacity it can memorize any given sample of the target representation and thus not achieve the desired results. To prevent this, we generate another pair of samples, $F(G(x)$ and $G(F(y))$, from previously generated samples $G(x)$ and $F(y)$. This restricts the target mapping space, allowing us to ensure that the "content" of the image stays similar. The loss function is defined as:

\begin{equation}
	\mathcal{L}_{cycle}(G, F) = \mathbb{E}_{x \sim p_{data}(\mathbf{A})}[\|F(G(x) - x\|_1] + \mathbb{E}_{y \sim p_{data}(\mathbf{B})}[\|G(F(y) - y\|_1]
\end{equation}

\paragraph{Total Loss} We now get the full equation of the loss function:

\begin{equation}
	\mathcal{L}_{total}(G, F, D_A, D_B) = V_{A \rightarrow B}(G, D_B) + V_{B \rightarrow A}(F, D_A) + \lambda L_{cycle}(G, F) \label{eq:cyclegan_total}
\end{equation}

where $\lambda$ is the weighting factor of the cycle loss, which empirically was selected to be 10. From this equation, we aim to solve:

\begin{equation}
	\min_{G, F} \max_{D_A, D_B} \mathcal{L}_{total}(G, F, D_A, D_B)
\end{equation}

\paragraph{Identity Loss \label{identity-loss}} Taigman et al. has discovered that introducing an additional loss, which tests the behaviour of the generators when given an input from the target distribution, increases in performance. When given such an input, we expect the generator to works alike the identity function. We define such loss as:

\begin{equation}
	\mathcal{L}_{idt}(G, F) = \mathbb{E}_{x \sim p_{data}(\mathbf{A})}[\|F(x) - x\|_1] + \mathbb{E}_{y \sim p_{data}(\mathbf{B})}[\|G(y) - y\|_1]
\end{equation}

CycleGAN allows for a style transfer without any paired dataset or pre-trained classifiers; therefore, I have chosen to investigate further possibilities of audio style transfer using the CycleGAN architecture.

## Audio Signal Processing Using Neural Networks

The neural network structures introduced above have been designed mainly for image style transfer; in order to adapt such methods for audio style transfers I will investigate on previous mechanisms for audio processing using neural networks in this section.

### Data Representations

We can choose either to use the raw audio directly as the input to our neural network model or convert it into some intermediate representation. Recent network models such as WaveNet do so, while previous audio processing networks use an intermediate representation in order to generate a 2D map of the audio input. Such mechanisms, called time-frequency analyses, generate a 2D matrix which shows how the frequencies content of the audio clip change over time. Among such representations we will look into Short Time Fourier Transforms (STFT) and Constant-Q Transforms (CQT). 

\paragraph{Short Time Fourier Transform} STFT, in our case discrete-time STFT (since we are transforming a sampled audio), transforms the audio file into a matrix $D$, where each $D_{mw_{k}}$ notate the magnitude and phase of the audio file at frequency bin $w$ and time step $m$. Such matrix is computed by dividing the audio file into constant sized windows (which is usually overlapped), as below:

\begin{equation}
	\mathbf{STFT}\{x[n]\}(m, w_k) = \sum_{-\infty}^{\infty} x[n]w[n - m]e^{-jw_kn}
\end{equation}

where $x[n]$ is the input signal and $w[n]$ the window function at the n^th^ window. The process can be thought of as computing the discrete Fourier transform on $x[n]w[n - m]$.

\subparagraph{Griffin-Lim} One advantage of using STFT is that we can recover the original audio from the STFT matrix using Griffin-Lim algorithm. The details of the Griffin-Lim algorithm is beyond the scope of this project; however, we will later use an implementation to recover an audio sample from the neural networks.

\paragraph{Constant-Q Transform} Constant-Q Transform is another time-frequency transform which unlike STFT does not use a fixed frequency window, but uses a exponentially spaced frequency bands. Frequency cutoffs are defined as $\omega_k = 2^{\frac{k}{b}} \omega_0$ for $k \in \{ 1, ..., k_{max}\}$, where $\omega_0$ is the starting frequency, $k_{max}$ the number of frequency bands to compute (i.e. the height of the resulting matrix), and $b$ is the scaling factor. Each frequency band $\Delta_k = \omega_k - \omega_{k-1} = \omega_k 2^{\frac{1}{b} - 1}$, which gives a constant frequency to resolution $Q = \frac{\Delta_k}{\omega_k}  = 2^{\frac{1}{b} -1}$. Resulting matrix reflect human perception of sound; when $b$ is set to 12, the frequency cutoffs match the spacing between each semitone in an equally tempered scale. 

\paragraph{STFT vs CQT} Since CQT uses a logarithmic scale of frequency (due to the exponential cutoffs) and has explicit parameters for starting and ending frequencies, we can ensure that all frequency ranges used by modern music. It also has a higher resolution on lower frequencies, which allows for a better representation of instruments in the lower frequencies. [@timbretron] Given that higher frequencies are less frequent in most songs, CQT seems to be the appropriate representation. However, CQT lacks a restoration algorithm unlike STFT, and therefore may be hard to use for audio transformation.

\paragraph{Raw audio vs Time-Frequency Analyses} Dieleman et al. compared performances of audio classification and auto-tagging tasks between raw audio and intermediate representations. While the performance of the classifiers using intermediate representations work better, they have also shown that neural networks using raw audio can also autonomously learn the features to a comparative degree.[@endtoend] WaveNet, which we will cover later in this section, uses raw audio as an input, and learns timbral qualities, such as human voice.[@wavenet]

\paragraph{Mel Frequency Scale} To mitigate the resolution limit in lower frequencies for STFT matrices, we scale the frequencies of the raw audio to a logarithmic scale before computing the STFT matrix. We use the follow equation to do so--given frequency $f$:

\begin{equation}
	m = 2595 \log_{10} (1 + \frac{f}{700})
\end{equation}


### WaveNet 

\begin{figure}[!h]
	\includegraphics[width=0.7\textwidth]{./figures/wavenet.png}
	\centering
	\caption{A variant of WaveNet Architecture, implemented by Nvidia} \label{fig:wavenet}
\end{figure}

Figure \ref{fig:wavenet} shows the architecture of WaveNet, introduced by Oord et al., which learns to generate raw audio. [@wavenet] WaveNet generates new sample points that are conditional to the outputs of the previous time step, i.e. the each element of the output waveform of length $T$,  $\mathbf{x} = \{x_0, ..., x_T\}$, is generated by the product of the probability distribution that is conditional to all previous time step:

\begin{equation}
	p(\mathbf{x}) = \prod_{t=0}^{T} p(x_t | x_0, ..., x_{t-1})
\end{equation}

Both the input and output of WaveNet are expected to be quantized, which allows the input sequence to be represented into a one-hot vector. The output of the examples are `softmax`ed to notate the probability of the position of the next sound sample.

\paragraph{Quantization} To quantize the audio input, we use $\mu$-law, which is traditionally used to quantize PCM signals (raw audio signal standard) for analog to digital conversion. Given an input sample point $x, 0 \leq x \leq 1$ and $\mu$ the quantization equation is:

\begin{equation}
	F(x) = sign(x) \frac{\log(1 + \mu \lvert x \lvert )}{1 +  \mu}
\end{equation}

$\mu$ notates the vector space to which the signal is quantized to; with $\mu = 255$, the resulting signal can be represented in $\{0, 1\}^{256}$.

\begin{figure}[!h]
	\includegraphics[]{./figures/causal_conv.png}
	\centering
	\caption{A diagram of a causal convolutional network.} \label{fig:causal_conv}
\end{figure}

\paragraph{Causal Convolutional Networks} The integral part of WaveNet is the Causal Convolutional Network. It is a series of convolutional neural networks with increasing dilation and left-oriented padding, to allow a binary-tree like input sequence, while only being dependent on previous sample points and not using strides or pooling. Causal Convolutional Networks allow for significantly large reception field with a handful of convolutional layers, which allow the network to capture lasting audio qualities such a timbre. 

\paragraph{Training and Inference} During training, the causal convolutional networks trains like any other convolutional network with the entire sample waveform as the input of the network. However, during inference we generate each sample point individually, through an iterative process of feeding the output of the network back into the network to get the next sample point. Such inference mechanism allow us to model the conditional probability $p(x_t | x_0, ..., x_{t-1})$ as discussed above.

\paragraph{Gated Activation} WaveNet uses two activation function $\tanh$ and sigmoid as shown in figure \ref{fig:wavenet}. By multiplying the results given by the two activation functions, the sigmoid function acts as gate to limit the effect of the result generated by the $\tanh$ activation. Such ideas have been borrowed from LSTM, a type of recurrent neural networks, and have been shown effective by PixelCNN. [@pixelcnn] @wavenet found that using gated activation unit increase the performance of WaveNet significantly.

\paragraph{Conditioning} Alike conditional adversarial networks, we can generate a WaveNet model that is conditional on an external input, i.e. model the $p(x_t | x_0, ..., x_{t-1}, c)$ given a conditional input $c$. Deep Voice uses such conditional model, using voice content as conditional input to generate a Text-to-Speech engine. We will use the conditional model to input pitch data, as we will see in the section below for TimbreTron and Universal Music Translation Networks.

### TimbreTron

\begin{figure}[h]
	\includegraphics[width=\textwidth]{figures/timbretron.tikz}
	\centering
	\caption{TimbreTron model architecture. CQT matrices are notated $CQT$ and the raw audio $PCM$ to distinguish the type of data generated.} \label{fig:timbretron}
\end{figure}

TimbreTron is a combination of CQT, CycleGAN, and WaveNet that has been shown to be effective for timbre transfer. Instead of images, it uses CQT matrix as an input to the generators for CycleGAN in order to transform it to that of a different instrument. The generated matrix is used as a conditional input to the WaveNet synthesizer, which generates a natural audio sample.

\paragraph{Full Spectrum Discriminator} The discriminator used by the original CycleGAN model, PatchGAN, computes the likelihood probability on 70x70 patches of the image and takes the mean of each patch loss. Since such discrimination process do not make sense when dealing with audio data, the discriminator has been modified to compute the entire frequency spectrum. The details of the discriminator design is not included in the paper, so in the implementation section I will present different discriminators of original design.

\paragraph{WaveNet} To mitigate the lack of a restoration algorithm for CQT, TimbreTron uses a conditional WaveNet to generate the audio from the CQT matrix. Although not specified in the paper, The CycleGAN model and WaveNet are assumed to be trained separately to yield a theoretically sound model. If so, it is questionable whether they needed to use CycleGAN to transform the CQT data to resemble that of the other instrument, as our objective can be simplified to extracting the pitch data from the CQT representation.

\paragraph{Gradient Penalty} TimbreTron uses the GAN value function introduced by the Wasserstein GAN model, along with gradient penalty as the Lipschitz constraint introduced by Gulrajani et al. The resulting value function given posterior distribution $p_g$ from the generator and random sample $p_z$ is:

\begin{multline}
	V(G, D) = \underbrace{
		\mathbb{E}_{x \sim p_{g}(\mathbf{x})} [D(x)] - \mathbb{E}_{x \sim p_{data}(\mathbf{x})} [D(x)]
		}_\textrm{Wasserstein GAN value function} \\ +
		\underbrace{
			\mathbb{E}_{x \sim p_z(\mathbf{x})} [(\| \nabla_x D(x) \|_2 - 1)^2]
		}_\textrm{Gradient Penalty}	
\end{multline}

which has been shown by @gp .

By using gradient penalty the authors of TimbreTron claim that they were able mitigate the unstable training behaviour shown by the model caused by the fast training speed of the discriminator.

\paragraph{Diminishing Identity Loss} To make sure the generators preserves the pitch content within the matrix during the initial part of the training process, TimbreTron introduces the identity loss (as reviewed in section \ref{identity-loss}). However, they decided to decrease the weighting of the identity loss throughout the training process since it can hinder the legitimate style transfer process as the training continues.

### Universal Music Translation Network

\begin{figure}[h]
	\includegraphics[width=\textwidth]{figures/umtn.tikz}
\centering 
\caption{Architecture of the Universal Music Translation Network \label{fig:facebook}}
\end{figure}

#### WaveNet AutoEncoder

![https://magenta.tensorflow.org/nsynth \label{fig:nsynth}](./figures/nsynth.png)

Engel et al. introduced an encoder network design which when combined with a conditional WaveNet decoder acts as an autoencoder, as shown in figure \ref{fig:nsynth}. The encoder is a chain of convolutional layers with increasing dilation and residual connections much like WaveNet, but with ReLU activations. The paper showed that the resulting vector $Z$ can encode temporal features. For Universal Music Translation Network, we aim to use this Temporal Encoder to extract pitch from the input audio. 


#### Domain Agnostic Neural Networks

\begin{figure}[h]
    \def\svgwidth{\textwidth}
    \input{dann.pdf_tex}
    \centering
	\caption{Structure of Domain Agnostic Neural Networks \label{fig:dann}}
\end{figure}

# Implementation

## Datasets 

\paragraph{Free Music Archive}

\paragraph{Youtube}

\paragraph{Maestro}

\paragraph{GuitarSet}

## Classifiers

## CycleGAN

TimbreTron?
- Uses CycleGAN on CQT spectrograms, and produces wavs by using WaveNet
- Architecture needs 6 networks! Too large to fit into a system
- We will look into possibilities of a Wavenet-less system


\paragraph{Distributed Model}

### Generator Models

- basic Conv2d 
- freq as channel

\paragraph{Discriminators} Shallow(Semi), Temporal(?), Full_spectrum (X)

## Universal Music Translation Network

- General Architecture

* Troubleshooting, and differences from paper.

## Original Model

STFT -> Resnet2d ->  WaveNet
				 ->  Classifier 

## Code Structure


# Evaluation


# Conclusion
