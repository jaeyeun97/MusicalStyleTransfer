
---
header-includes: 
- \usepackage{tikz}
- \usepackage{tikzscale}
- \tikzset{every picture/.style={line width=0.75pt}}
subparagraph: True
numbersections: true
documentclass: report
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
	\includegraphics[width=\textwidth]{./figures/conv.tikz}
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
	\includegraphics[width=\textwidth]{figures/pix2pix.tikz}
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
	\includegraphics[width=\textwidth]{figures/cyclegan.tikz}
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


\paragraph{Constant-Q Transform} Constant-Q Transform is another time-frequency transform which unlike STFT does not use a fixed frequency window, but uses a exponentially spaced frequency bands. Frequency cutoffs are defined as $\omega_k = 2^{\frac{k}{b}} \omega_0$ for $k \in \{ 1, ..., k_{max}\}$, where $\omega_0$ is the starting frequency, $k_{max}$ the number of frequency bands to compute (i.e. the height of the resulting matrix), and $b$ is the scaling factor. Each frequency band $\Delta_k = \omega_k - \omega_{k-1} = \omega_k 2^{\frac{1}{b} - 1}$, which gives a constant frequency to resolution $Q = \frac{\Delta_k}{\omega_k}  = 2^{\frac{1}{b} -1}$. Resulting matrix reflect human perception of sound; when $b$ is set to 12, the frequency cutoffs match the spacing between each semitone in an equally tempered scale. 

\subparagraph{Griffin-Lim} We can recover the original audio from the STFT matrix using Griffin-Lim algorithm. The details of the Griffin-Lim algorithm is beyond the scope of this project; however, we will later use an implementation to recover an audio sample from the neural networks.

\paragraph{STFT vs CQT} Since CQT uses a logarithmic scale of frequency (due to the exponential cutoffs) and has explicit parameters for starting and ending frequencies, we can ensure that all frequency ranges used by modern music. It also has a higher resolution on lower frequencies, which allows for a better representation of instruments in the lower frequencies. [@timbretron] Given that higher frequencies are less frequent in most songs, CQT seems to be the appropriate representation. 

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
	F(x) = sign(x) \frac{\log(1 + \mu \lvert x \lvert )}{1 +  \mu} \label{eq:mulaw}
\end{equation}

$\mu$ notates the vector space to which the signal is quantized to; with $\mu = 255$, the resulting signal can be represented in $\{0, 1\}^{256}$.

\begin{figure}[h]
	\includegraphics[width=\textwidth]{./figures/causal_conv.png}
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

\paragraph{WaveNet} TimbreTron uses a conditional WaveNet to generate the audio from the CQT matrix. Although not specified in the paper, The CycleGAN model and WaveNet are assumed to be trained separately to yield a theoretically sound model. If so, it is questionable whether they needed to use CycleGAN to transform the CQT data to resemble that of the other instrument, as our objective can be simplified to extracting the pitch data from the CQT representation.

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
	\includegraphics[width=\textwidth]{figures/facebook_umtn.eps}
    \centering
	\caption{Universal Music Translation Network Diagram, as shared by Mor et al. \label{fig:facebook_umtn}}
\end{figure}

The Universal Music Translation Network (UMTN) by Mor et al. is another method for timbre transfer which uses the temporal encoder introduced by Engel et al. for Google Magenta project. UMTN extracts the pitch data from the source data using said encoder, and provides it as the conditional input to the WaveNet module. The paper states that such method has been effective for timbre transfer.

\begin{figure}[h]
	\includegraphics[width=\textwidth]{./figures/nsynth.png}.
	\centering
	\caption{https://magenta.tensorflow.org/nsynth \label{fig:nsynth}}
\end{figure}

#### The Temporal Encoder

Engel et al. introduced an encoder network design which when combined with a conditional WaveNet decoder acts as an autoencoder, as shown in figure \ref{fig:nsynth}. The encoder is a chain of convolutional layers with increasing dilation and residual connections much like WaveNet, but with ReLU activations. The paper showed that the resulting vector $Z$ can encode temporal features. For UMTN, we aim to use this Temporal Encoder to extract pitch from the input audio. 

One integral design decision made for UMTN is that the encoder is shared by all inputs to the network model, while they train a WaveNet decoder for each instrument domain. This domain-independent encoder allows timbre transfer for audio input of an instrument not seen during the training process.

\begin{figure}[h]
	\includegraphics[width=\textwidth]{figures/dann.eps}
    \centering
	\caption{Structure of Domain Adversarial Neural Networks \label{fig:dann}}
\end{figure}

#### Domain Confusion

To further aid the domain independence of the output of the temporal encoder, the model includes a domain confusion network. It takes the output of the temporal decoder as the input and tries to predict its source domain. By training to maximise the loss function instead, as shown effective by Domain Adversarial Training of Neural Networks by Ganin et al., we train the encoder to output the pitch data regardless of the domain specific features (i.e. timbre). [@dann] @umtn does not mention how they have implemented their domain confusion but cites the @dann instead. Therefore I have assumed here that they have used the domain classifier as shown in figure \ref{fig:dann}, as that was the original design of @dann.

#### Pitch Shift

Lastly, to improve the generalization capability of the encoder (i.e. to translate well to unseen pitches in the dataset), all input audio file has been pitch-shifted, resulting in a slightly off-tune input. The length and amount of pitch shift is chosen randomly; details of the pitch shift will be described in the implementations section. 

#### Total Loss Function

Combining all elements detailed above, we can now write the loss functions of the model. Given 

* the encoder function $E$, 
* decoder function $D_i$ for each instrument $i \in I$,
* domain confusion network function $C$,
* input augmentation $O$, 
* random noise distribution $p_z$ 

we have the reconstruction loss:

\begin{multline}
	\mathcal{L}_{recon} = \sum_{i \in I} \mathbb{E}_{x \sim p_{data}(x), \{r, s, t, u\} \sim p_{z}} \\
	[\mathcal{L}_{decoder}(D_i(E(O(x, r), s), t))  - \lambda \mathcal{L}_{dc}(C(E(O(x, r), s), u))]
\end{multline}

where $\mathcal{L}_{decoder}$, $\mathcal{L}_{dc}$ are Cross Entropy loss functions.


# Implementation

The networks shared above is the current state of art in audio processing using neural networks. In this section I will implement network architectures inspired by these networks. I have used `PyTorch` to implement all models and data processing, and `librosa` to preprocess the audio samples. [@pytorch, @librosa] All hyper-parameters and preprocessing options are passed in through command-line options to allow for more flexible and faster experiment iterations. I have used code written by Zhu et al. [@cyclegan, @pix2pix] for CycleGAN image style transfer as the boilerplate of the project, but has been heavily modified to accommodate for flexible dataset combinations, diverse neural network models, and distributed model training (i.e. training a single model across multiple GPUs).

## Datasets 

The data loading module is split into two processes: a set of dataset-specific loaders that is responsible for selecting the right subset of the dataset using metadata files and commandline options, and a preprocessing pipeline, which applies the same preprocessing pipeline to all data samples. 

### Dataset Loader

The dataset loader is expected to:

* maintain a set of paths to audio samples,
* store the total size and relevant metadata such as duration,
* load the audio files from the dataset using `librosa`,
* resample to a user-set sample rate,
* and slice them into fixed intervals.

The `Dataset` module from PyTorch allows all audio slices are loaded parallelly, which reduces the latency caused by preprocessing in between training iterations. The original codebase includes a dynamic class loader for datasets that allows switching between datasets convenient. The code for the loader was modified to be able to combine (`zip` in Python terms) two different datasets into a set of pairs of data samples, instead of writing a loader for each combination of datasets. During training, the data loader randomizes the order of the samples, which prevents the models from training a mapping between the two pairs.

It is important to note that the batch of data generated by the loader is always balanced, since we provided an equal number of samples from each domain on every iteration.

### Sources

First, I gathered a diverse set of datasets in order to try different combinations of audio style transfer; I tried to gather a dataset of recorded music as opposed to MIDI generated data, in order to encapsulate real world examples of music which contain different playing styles of instruments.

\paragraph{Free Music Archive} The Free Music Archive has a set of license-free music, classified by genre. Defferrard et al. compiled the songs from the archive into 30 second clips along their metadata information organized into a csv file. [@fma]

\paragraph{Youtube} Thanks to `MellowBeatSeaker`, a Korean YouTube channel that streams "Chill Lo-Fi Study Beats", I could gather 60 hours of curated chill music. These files were processed using `librosa` to be cut into non-silent intervals.

\paragraph{Maestro} Maestro is a dataset of piano music compiled by the Google Magenta Project. It includes 172 hours worth of MIDI and wav files. They were gathered by using Yamaha Disklavier, which can record MIDI data from acoustic piano keys. Only the audio data has been used for this project. [@maestro]

\paragraph{GuitarSet} GuitarSet is compiled by Xi et al. at NYU Music and Audio Research Lab using hexaphonic pickups and microphones on acoustic guitars. The dataset contains pairs of soloist and accompanist recordings, which are mixed using `numpy` when the dataset is loaded. [@guitarset]

### Preprocess 

The preprocessing pipeline processes the sliced data samples into the necessary form needed for different models. Each model sets the default preprocessing options to indicate necessary elements, and the data samples are processed "to spec" before being converted into `PyTorch` tensors. I used `librosa`, a python audio processing package, to convert the data samples into Mel frequency scale and generate STFT or CQT matrices. 

If the model uses STFT or CQT, the output matrix is separated into magnitude and phase by calculating the absolute value and the angle of the complex value. The magnitude is then $\log$ed and linearly scaled to have a maximum of 1 and minimum of -1. This process, as done by GANSynth and DeepVoice, provides a wider variance and prevents gradient explosion from large input values. Furthermore, to remove any volume differences between the samples when not using time-frequency analyses, I have normalized the volume of all audio samples used for the datasets using `ffmpeg-normalize` prior to any training. 

Quantization and pitch shift, which are needed for UMTN and other WaveNet based models, are part of the preprocessing pipeline as well. Pitch shift is done by `librosa`, on a single random-length interval, chosen within the 25% to 75% mark with respect to the length of the data sample. The amount to be shifted was sampled from uniform distribution between 0 and 1. $\mu$-law function needed for quantization was implemented based on equation \ref{eq:mulaw}.

#### Postprocess 

The parameters that are needed to recover the original matrix are given to the network model with the data samples to be passed on to the postprocess pipeline. Inverses of all preprocess functions are implemented. STFT and CQT matrices are reconstructed using the Griffin-Lim algorithm which has been implemented by `librosa`. Its implementation of the restoration algorithm for CQT matrices, however, is unstable and the authors of the library do not recommend using it for purposes other than that diagnostic applications. With a simple experiment of applying CQT and restoring it, with $b = 120$ and $k_max = 84$, I could verify that the algorithm can successfully restore the audio for the purpose of this project.

## Network Models

In this section I introduce two neural network models inspired by TimbreTron and UMTN, along with an implementation of UMTN. TimbreTron could not be implemented because the network model was too large to fit into my system; moreover, @timbretron does not include many details (such as the generator architecture) which is essential to the implementation of the network.

First of the two new models is the CycleGAN model that uses the Griffin-Lim algorithm as the reconstruction mechanism. I tried to implement different generator and classifier networks to use with the time-frequency representations. Unlike generator networks, classifier networks can be evaluated independently in order to test the capacity of the network; hence I have written a separate model to evaluates these networks.

The second model was inspired by UMTN; while the UMTN learns pitch data from the temporal classifiers, I wondered if the results could change by using an encoder network with a time-frequency representation. The encoder networks from the CycleGAN model were used to implement such a model.

### Classifiers

I have experimented with three different classifier networks. First is `PatchGAN` which has been introduced by the original CycleGAN paper. The other two are of my original design, for which I will explain the intuition and implementation in the paragraphs below. 

#### PatchGAN classifier (Baseline)

#### `conv1d`

Failed to converge

#### Timbral classifier

### CycleGAN

@timbretron introduces a model that uses Griffin-Lim on STFT matrices as the baseline, but fails to present the equivalent for CQT representations. By implementing a CycleGAN model that uses Griffin-Lim reconstruction on both STFT and CQT representations, I hoped to clarify (1) the necessity of WaveNet in the TimbreTron model, and (2) the differences between STFT and CQT CycleGAN models.

#### Generator Models


#### Discriminators 

For discriminator units, we use the classifiers that have been evaluated above in section \ref{classifiers}.


#### Distributed Training System

\begin{figure}[h]
	\includegraphics[width=\textwidth]{./figures/cyclegan_gpu.tikz}
	\centering
	\caption{A modification of figure \ref{fig:cyclegan} to show how the model is split between the two GPUs. The two yellow points show the point at which the tensor has to be transferred; this way we can minimize the necessity of transferring the output vector.} \label{fig:cyclegan_gpu}
\end{figure}

Some generator models made the entire network architecture too large to fit in the memory of a single GPU. The original model implemented by Zhu et al., however, only had multi-GPU training capacities using dataset distribution. Therefore, the code structure was modified to store a generator and a discriminator pair on each GPU, and the relevant device management code was modified. With distributed modelling techniques, the transfer of tensors between the GPU can become a major bottleneck within the system; therefore, the modified codebase relies on individual model implementation to design ways to transfer minimal information between the devices. For the CycleGAN module, this was done as seen in figure \ref{fig:cyclegan_gpu}, which needs only two transfer between the devices for a single iteration. `PyTorch` handles the transfer of gradients between devices during back-propagation so there was no worry of losing gradient information.

### TimbreTron

With the distributed training in place, I was hoping to be able to train TimbreTron on this system; however, there was still not enough memory resources to train the six networks together. Without looking further ways to solve this issue, I have decided to implement the Universal Music Translation Network which included some promising audio samples within the paper.

### Universal Music Translation Network

\begin{figure}[h]
	\includegraphics[width=\textwidth]{figures/umtn.tikz}
\centering 
\caption{Architecture of the Universal Music Translation Network \label{fig:umtn}}
\end{figure}

I don't know what the problem is here..


### Original Model

STFT -> Resnet2d ->  WaveNet
				 ->  Classifier 

## Code Structure



# Evaluation


# Conclusion
