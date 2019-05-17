---
header-includes: 
- \usepackage{tikz}
- \usepackage{tikzscale}
- \usepackage{multirow}
- \usepackage{subcaption}
- \tikzset{every picture/.style={line width=0.75pt}}
- \DeclareMathOperator*{\argmax}{arg\!\max}
- \DeclareMathOperator*{\argmin}{arg\!\min}
- \setcounter{secnumdepth}{3}
subparagraph: True
documentclass: report
geometry:
- margin=2cm
papersize: a4
fontsize: 12pt
bibliography: diss.bib
csl: apa.csl
nocite: |
	@*
---

\pagestyle{empty}

\rightline{\LARGE \textbf{Charles Jae Yeun Yoon}}

\vspace*{60mm}
\begin{center}
\Huge
\textbf{Musical Style Transfer using Neural Networks} \\[5mm]
Computer Science Tripos -- Part II \\[5mm]
King's College \\[5mm]
\today  % today's date
\end{center}

\newpage

\section*{Declaration}

I, Charles Jae Yeun Yoon of King's College, being a candidate for Part II of the Computer Science Tripos, hereby declare that this dissertation and the work described in it are my own work, unaided except as may be specified below, and that the dissertation does not contain material that has already been used to any substantial extent for a comparable purpose.

\bigskip
\leftline{Signed}

\begin{figure}
	\includegraphics{/home/jaeyeun/Documents/sign.png}
\end{figure}

\medskip
\leftline{Date \today}

\newpage

\chapter*{Proforma}

\begin{tabular}{ll}
Candidate Number:   & \bf 303488330                    \\
Project Title:      & \bf Musical Style Transfer using Neural Networks \\
Examination:        & \bf Computer Science Tripos -- Part II, June 2019 \\
Word Count:         & \bf 11587 \footnotemark[1] \\ 
Project Originator: & Prof. Alan Blackwell                      \\
Supervisor:         & Dr. Andrea Franceschini                    \\ 
\end{tabular}

\footnotetext[1]{This word count was computed
by \texttt{texcount diss.tex}
}
\stepcounter{footnote}

\section*{Original Aims of the Project}

The Original Aims of the project was to:

1. compile a dataset of musical audio samples;
2. implement an effective audio feature classifier;
3. test the classifier with scientific methods and verify its performance;
4. design and implement neural network architectures that perform musical style transfer;
5. and conduct a human evaluation experiment to evaluate the neural network.

\section*{Work Completed}

All five of the original aims of the project have been completed with successful results. Four datasets of different musical characteristics were used, where as one of the datasets were compiled by the dissertation author, with permission from the artists and curator. Three models of classifiers were implemented, where two of them showed successful behaviour during testing. Three different neural network architectures were implemented, where one of the models showed successful statistical behaviour over a dataset pair between piano and guitar samples. Human perception experiment was conducted successfully with the audio style transfer from guitar samples to piano samples especially doing well. 

\section*{Special Difficulties}

No special difficulty has occurred during the duration of this project.
\tableofcontents

\listoffigures

\newpage

\section*{Acknowledgements}

The author of this dissertation would like to thank:

* `MellowBeatSeeker`, for offering his playlist of music to be used for a dataset of this project,
* Prof. Alan Blackwell and the Rainbow Laboratory, for providing the necessary hardware for this project,
* Matthew Else, for offering additional hardware resources for this project.



# Introduction 

Software instruments became an integral part of recent music production, by allowing musicians to play a variety of instruments using a single MIDI controller. These instruments are essentially a collection of pre-recorded samples of each pitch, which are played back to the user given the input of the controller. These controllers, which usually have a piano-like interface, takes pitch and modulation information from the user, and relays this information to the software instrument. The virtual instrument, when given input, generates the sound of the instrument it is modelled after with audio processing techniques to control certain elements that are not present in the MIDI command sequence. We can, therefore, think of these instruments as a certain quality of sound, which when given a MIDI input, generates music.

Identically, we can consider any sequence of musical sound as an addition of a set of qualities, such as pitch, timbre, and rhythm, that make that sequence unique from other sounds. The aim of this project is to find a computational mechanism to recognize such qualities, and generate or transform a given sound sample to imitate that quality of sound. 

\paragraph{Why Neural Networks?} Human beings, when given a sample of sound, can recognize such features however hard it may be to define the quality that they recognize. It is difficult, however, to use traditional computational algorithms to perform such tasks, even given ample set of domain knowledge relating to audio signal processing. Neural Networks, over the past decade have shown to be effective in such areas, allowing complex image recognition and transformation to be possible. Therefore, I have implemented a neural network model that performs a similar task with audio signals.

\paragraph{Classes of audio features} In order to perform such a task, I will be defining sets of audio samples that share a similar quality of sound. For example, a set of piano recordings share the timbre of the instrument. Moreover, there are classes of such features of sound, e.g. the quality "sound of a piano" is a timbre. Some of these class of audio characteristics, like genre, are harder to define. However, it may be hard to define what classical music is, but humans can recognize and classify classify classical music from the rest. Likewise, any curated set of music compiled with a certain theme can also be considered an audio feature. We will be looking at two different classes of qualities that are at different points in the spectrum of definability; namely, timbre and "chillness". "Chillness" is a difficult concept to define, like the other audio qualities above; however, a plethora of playlists describe their compilation to be "chill", and a number of people agree and search for this quality. Since there is a Korean YouTube channel `MellowBeatSeeker` that streams "Chill Study Beats," which is streamed by more than a thousand people regularly, we will use their data as the standard to chillness.
\
\paragraph{Intuition} Let us say that we have a set of piano recordings, $P$, and a set of guitar recordings, $G$. $P$ has a common characteristic $t_p$ and $G$, $t_g$, where $t_p$ and $t_g$ are of the same class, Timbre. We now define a transformer function $T: P \rightarrow G$, which transforms piano music into guitar music. This is illustrated in Figure \ref{fig:t}.

\begin{figure}
	\includegraphics{./figures/sound_t.tikz}
	\centering
	\caption{The audio style transformer $T$} \label{fig:t}
\end{figure}

The objective of $T$ is to discriminate $t_p$ and $t_g$, and when given a sample $p \in P$, modify $t_p$ into $t_g$ from $p$. Our first goal is to model this transformer $T$ using a neural network.

For the case of "chillness", the transformation is not as straightforward, and there are multiple ways to models "chillness". First, we have to decide if "chillness" is a measure in continuous scale, or a characteristic that an audio sample can either possesses or not. Since, it is difficult to acquire a dataset labelled with a "chillness" measure, we assume the latter. Then, we have to decide whether the quality of being "not chill" is a separate quality of its own. If so, we can model the transformation function as we have for timbre. If not, we would model a transformation function $R: C^\complement \rightarrow C$, where $C$ is the set of chill music. Previous literature makes the first approach more available, hence we define chillness characteristic $t_c$ and its complement $t_{\lnot c}$, and approach it in a similar way to timbre. 

From here on, we divide the qualities of sound into two categories: content and style. Contents are the qualities that we do not want to modify, such as pitch. Styles are the other qualities that is not the content of audio, which we want to modify.

# Preparation

A number of previous work have been done on the subject of audio style transfer; however, in order to better understand those works, a number of concepts in machine learning need to be reviewed. Although Neural Networks are introduced in *Part IB Artificial Intelligence* and *Part II Machine Learning and Bayesian Inferences*, many of the contents of the project are beyond the coverage of Tripos courses; therefore, in this chapter I aim to clarify the starting point and the previous works I relied on to implement the project.

## Starting Point

* Machine Learning and Deep Neural Networks: Other than the introduction provided by the Part IB course, I was only exposed to neural networks from a previous UROP project. In order to strengthen foundational knowledge on theories and foundations of machine learning and neural networks, a number of textbooks and papers were reviewed and summarized below. I had experience with `PyTorch` from previous projects, so I chose it for alacrity of development. 

* Audio Processing: I had little experience with Digital Signal Processing; although most material that will be introduced below require little domain knowledge, I reviewed the course materials for the Part II Digital Signal Processing course.

* Miscellaneous Tooling: Jupyter Notebook/Lab and Git were used for training monitoring and version control.

## Machine Learning and Neural Networks 

Machine Learning generally can be classified into a number of tasks; for this project we are mainly concerned with supervised learning.

Each entry in the dataset is a collection of features describing it; for example, an entry from a dataset generated by a sensor network would include each sensor reading as a feature. While each data point can be of any size, we can think of each as a single vector without loss of generality.

For supervised learning, there is a label or a result that is associated with each entry of inputs. We assume that each of these output values has been generated from some probability distribution that is conditional on the input features. Then the task of a supervised learning algorithm is to find that distribution; given probability of the input $P(x_1, ..., x_n)$, and output probability of $P(y)$, the task at hand is to find the conditional probability $P(y \mid x_1, ..., x_n)$. This is analogous to defining a function (a 'hypothesis') $h: X^n \rightarrow Y$ that maps samples of the input distribution and to those of the output distribution, i.e. "learning" the transformation between the distributions. If the output space $Y$ is continuous, our task is regarded as regression. If $Y$ is a set of finite categories, we are doing classification.

However, it is evident that we cannot observe all factors that contribute to the output $y$, so some models model this uncertainty by adding a random noise variable $z$, i.e. $P(y) = P(y \mid \vec{x}, z) P(\vec{x} \mid z) P(z)$. We further assume that the observations are independent from the said uncertainties, i.e. $P(\vec{x} \mid z) = P(\vec{x})$, which yields $P(y) = P(y \mid \vec{x}, z) P(\vec{x}) P(z)$. To draw the analogy of functions, The conditional probability $P(y \mid \vec{x}, z)$ can be thought of the function $h: X^n \times Z \rightarrow Y$ where $Z$ is the set of samples from the noise distribution.

In any case, the function $h$ can be represented by a collection of weights applied to each feature, $\vec{w}$; to make this clear, we notate $h$ as $h_{\vec{w}}$. Our task then is to find the appropriate weights that computes $h$ well given the input distribution. For this project, we do so by using multilayer neural networks.

\begin{figure}[h]
	\begin{subfigure}{.5\textwidth}
		\includegraphics[width=\textwidth]{./figures/perceptron.tikz}
		\centering
		\caption{A single perceptron} \label{fig:perceptron}
	\end{subfigure}
	\begin{subfigure}{.5\textwidth}
		\includegraphics[width=\textwidth]{./figures/simple_neural_network.png}
		\centering
		\caption{A fully connected Neural Network}
	\end{subfigure}
\end{figure}

### Neural Networks 

First we define a single node of a neural network, called a perceptron. A perceptron is a simple linear operation usually depicted as Figure \ref{fig:perceptron}. Given the input $\vec{x}$, weights $\vec{w}$, and a bias input of $\vec{b}$ (which is usually randomly initialized), it outputs a single scalar value $\vec{w}^\intercal x + \vec{b}$. A single layer of a neural network is a set of these perceptrons, which results in a vectorized output. The layer of perceptrons can be represented as the weight matrix $W$ where $W = (\vec{w_0} ... \vec{w_n})^\intercal$, where $n$ is the number of layers, so that the output vector $\vec{y} = W \vec{x}$ where $y_i = \vec{w_i}^\intercal \vec{x}$. A neural network is a series of these layers applied sequentially. The process of computing each layer to get an output is called feedforward, and such networks are either called feedforward neural networks or fully connected networks.

\paragraph{Activation Functions} Since the output of each perceptron is linear the output of the neural network described above can only model a linear function on $\vec{x}$. To be able to model non-linear function, we apply an activation function to the output of each perceptron. Sigmoid, Rectified Linear Unit (ReLU), and $\tanh$ are the popular choices for this purpose. For more information on the class of function a neural network can approximate, refer to the Universal Approximation Theorem. 

\begin{align}
	\sigma(x) &= \frac{1}{1 + e^{-x}} \label{eq:sigmoid} \\
	ReLU(x) &= \begin{cases}
		x & x \geq 0 \\
		0 & x < 0
	\end{cases}	
\end{align}

The sigmoid function, shown by Equation \ref{eq:sigmoid} has a range of (0, 1), which make it ideal for binary classification. Similarly, $\tanh$ is used to restrict the output to (-1, 1). ReLU is a common activation function chosen for deeper neural networks, since its derivative is simple, which helps the calculation of gradients, as will be discussed in Section \ref{back-propagation} below. 


### Loss Functions

As stated before, our objective is to find the appropriate set of weights that can successfully model the output distribution. In order to do so we "fit" the network outputs to the observed outputs, or adjust the weights to maximize the probability of getting the observed output given the observations. To do so effectively, we need a heuristic which evaluates the efficacy of the weights, which is usually calculated as some calculation of distance between the current outputs and the observed outputs. We call these heuristic functions *loss functions*, and I will introduce few popular loss functions.

\paragraph{Mean Squared Error (MSE)} The mean squared error loss function, as shown by Equation \ref{eq:mse}, is a straightforward loss function that uses the squared difference, which is always greater than 0 and has a minima of 0 when $\vec{y}_i = h_{\vec{w}}(X_i)$ for all $i$ and thus fits the criteria of a heuristic function. It is widely used as the loss function for many regression and classification tasks. However, the MSE loss function heavily weighs the outliers, which makes it unsuitable for some models.

\begin{equation}
	\mathcal{L}_{MSE}(\vec{w} \mid X, \vec{y}) = \frac{1}{N} \sum_{i = 0}^{N} (\vec{y}_i - h_{\vec{w}}(X_i))^2 \label{eq:mse}
\end{equation}

\paragraph{Negative Log Likelihood (NLL)} The likelihood is the probability of observing the observed output, given a particular weight and the observations, i.e.

\begin{align}
	\mathcal{L}_{likelihood}(\vec{w} \mid X, \vec{y}) &= P(\vec{y} \mid X, \vec{w}) \\
	&= \prod_{i = 0}^{n} P(\vec{y}_i |  X_i, \vec{w}) \label{eq:nllprod}
\end{align}

where $X$ is the entire feature observation, $\vec{y}$ the output observations, $\vec{w}$ the weights, and $n$ the length of the dataset. Note that we get Equation \ref{eq:nllprod} from assuming that each entry in the dataset is independent from each other. We cannot use the likelihood as the loss function, however, since (1) we want a function that we can minimize, and (2) the product of probability of all dataset is difficult to compute and is most likely to underflow. Since our objective is to get the weight $\vec{w} = \displaystyle \argmax_{\vec{w}} \mathcal{L}_{likelihood}(\vec{w} \mid X, \vec{y})$, we use the fact that

\begin{align}
	\argmax_{\vec{w}} \prod_{i = 0}^{n} P(\vec{y}_i |  X_i, \vec{w}) 
	&= \argmax_{\vec{w}} \log (\prod_{i = 0}^{n} P(\vec{y}_i |  X_i, \vec{w})) \\
	&= \argmax_{\vec{w}} \sum_{i = 0}^{n} \log P(\vec{y}_i |  X_i, \vec{w})
\end{align}

which holds since the maximum of a function is the maximum of its logarithm. To make it into a minimization problem, we invert the sign of the function, which yields the negative log-likelihood:

\begin{align}
	\mathcal{L}_{NLL}(\vec{w} \mid X, \vec{y}) &= - \sum_{i = 0}^{n} \log P(\vec{y}_i |  X_i, \vec{w}) \label{eq:nll1} \\
	&= - \sum_{i = 0}^{n} \log P(\vec{y}_i = h_{\vec{w}}(X_i)) \label{eq:nll2}
\end{align}

The NLL solves both of the problem that the likelihood function had earlier, and now can be used as a loss function. Equation \ref{eq:nll2} is equivalent to \ref{eq:nll1} since the prediction made given $X$ and $\vec{w}$ is $h_{\vec{w}}(X_i)$. In other words, our objective when minimizing the NLL Loss is maximizing the probability that $\vec{y}_i = h_{\vec{w}}(X_i)$.

\paragraph{Cross Entropy Loss} In Information Theory, the cross entropy between two probability distributions $p$ and $q$ is the amount of information (quantified as average number of bits) needed to recover the true distribution $p$ if the information is encoded in distribution $q$. It is defined as $H(p, q) = \mathbb{E}_p[- \log q]$, which for discrete number of examples with support $\chi$ is $- \sum_{x \in \chi} p(x) \log q(x)$. For machine learning applications, $q$ is the posterior distribution suggested by the hypothesis, and therefore can be expressed as $q(\vec{x} \mid \vec{w})$. Given that the output of the hypothesis is a valid class probability (which adds up to 1), minimizing the NLL Loss is equivalent to minimizing the cross entropy. For multi-class classification, which generates a vectorized output, we use the `softmax` function to restrict the output to valid probability space. Cross Entropy Loss from here on will refer to the composition of `softmax` and NLL Loss.


### Optimization

Given the loss function, we need a mechanism to optimize the weight so that the loss decreases. We do so by using an optimizer, in this case gradient descent, which updates the weights using the following equation:

\begin{equation}
	\vec{w}_{i_{t+1}} = \vec{w}_{i_{t}} + \lambda \frac{\partial \mathcal{L}(\vec{w})}{\partial \vec{w}_i} \label{eq:opt}
\end{equation}

Other mechanisms for iteratively updating weights for loss minimizations exist. The Adam optimizer, for example, is a popular choice for optimizing deep neural networks. The authors of Adam claim that:

> The method is straightforward to implement, is computationally efficient, has little memory requirements, is invariant to diagonal rescaling of the gradients, and is well suited for problems that are large in terms of data and/or parameters. The method is also appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients. [@Kingma2015AdamAM]

These characteristics make it a popular choice for training complex networks. We will also be using the Adam optimizer for this project.

### Back-propagation

Equation \ref{eq:opt} shows that we need to calculate the partial derivative of the loss function with respect to each weight values. In a neural network, such is calculated by a series of chain rules that includes all node within the path between the weight and the loss function. Formally, let:

* the error function 
* the output of the node $k$ in layer $l$, before the activation, be defined as $a^l_k$;
* the output of the node $k$ in layer $l$ be defined as $o^l_k = f(a^l_k)$ where $f$ is the activation function;
* the weight of $o^l_j$ to the node $i$ in the next layer be notated as $\vec{w}^l_{i, j}$;
* the bias of layer $i$ be notated $\vec{w}^l_{0,j}$ by defining $o^l_0 = 1$;
* the output of the neural network $\hat{y} = h_{\vec{w}}(\vec{x})$, where $\hat{y}_i = o^n_i$;
* the loss function $\mathcal{L}(\vec{w})$ be equivalently notated as $E(\hat{y})$;
* the number of layers be $n$;
* the number of perceptrons in layer $l$ be $r_l$.

Our objective is to calculate the partial derivative for all $i$, $j$:

\begin{equation}
	\frac{\partial \mathcal{L}(\vec{w})}{\partial \vec{w}^l_{i, j}} = \frac{\partial E(\hat{y})}{\partial a^l_i} \frac{\partial a^l_i}{\partial \vec{w}^l_{i, j}}
\end{equation}

Let the first term $\frac{\partial E(\hat{y})}{\partial a^l_i}$ be defined as the error $\delta^l_i$. The second term  $\frac{\partial a^l_i}{\partial \vec{w}^l_{i, j}}$ on the other hand can be calculated easily:

\begin{equation}
	\frac{\partial a^l_i}{\partial \vec{w}^l_{i, j}} 
	= \frac{\partial}{\partial \vec{w}^l_{i, j}} (\sum_{k=0}^{r_{l-1}} \vec{w}^{l}_{j, k} o^{l-1}_k)
	= o^{l-1}_j
\end{equation}

Therefore the objective can be notated as:

\begin{equation}
\frac{\partial E(\hat{y})}{\partial \vec{w}^l_{i, j}} = \delta{d^l_i} o^{l-1}_j \label{eq:grad}
\end{equation}

The calculation of $\delta^l_i$ depend on the layer of neural network. For the output layer,

\begin{equation}
	\delta^n_i = \frac{\partial E(\hat{y})}{\partial a^n_i} = \frac{d E(\hat{y})}{d \hat{y}} \frac{\partial \hat{y}}{\partial a^n_i}
\end{equation}

We assume that we can calculate derivative $\frac{d E(\hat{y})}{d \hat{y}}$ which is specific to the loss function. Additionally, the partial derivative:

\begin{equation}
\frac{\partial \hat{y}}{\partial a^n_i} 
= (\frac{\partial \hat{y}_0}{\partial a^n_i} \quad ... \quad \frac{\partial \hat{y}_{r_n}}{\partial a^n_i})^\intercal
= (\frac{\partial o^n_0}{\partial a^n_i} \quad ... \quad \frac{\partial o^n_{r_n}}{\partial a^n_i})^\intercal
\end{equation}

where $\frac{\partial o^n_j}{\partial a^n_i} = f'(a^n_i)$ if $i = j$ and is 0 otherwise. Therefore, $\delta^n_i$ is a vector, where:

\begin{equation}
	{\delta^n_i}_{j} = \begin{cases}
		f'(a^n_i) & i = j \\
		0 & \text{otherwise}
	\end{cases} \label{eq:bp_output}
\end{equation}

On the case of the hidden layers, 

\begin{equation}
	\delta^l_i
	= \sum_{j=0}^{r_{l+1}} \frac{\partial E(\hat{y})}{\partial a^{l+1}_j} \frac{\partial a^{l+1}_j}{\partial a^{l}_i} 
	= \sum_{j=0}^{r_{l+1}} \delta^{l+1}_j \frac{\partial a^{l+1}_j}{\partial a^{l}_i}
\end{equation}

where $\frac{\partial a^{l+1}_j}{\partial a^{l}_i}$ can be calculated as:

\begin{equation}
	\frac{\partial a^{l+1}_j}{\partial a^{l}_i}
	= \frac{\partial}{\partial a^{l}_i} (\sum_{k=0}^{r_{l+1}} \vec{w}^{l+1}_{j, k} o^{l}_k)
	= \frac{\partial}{\partial a^{l}_i} (\sum_{k=0}^{r_{l+1}} \vec{w}^{l+1}_{j, k} f(a^{l}_k))
	= \vec{w}^{l+1}_{j, i} f'(a^{l}_i)
\end{equation}

which finally yields:

\begin{equation}
\delta^l_i = f'(a^{l}_i) \sum_{j=0}^{r_{l+1}} \delta^{l+1}_j \vec{w}^{l+1}_{j, i}  \label{eq:bp_hidden}
\end{equation}

Using equations \ref{eq:grad}, \ref{eq:bp_output}, and \ref{eq:bp_hidden}, we can calculate all the partial derivative needed for optimization. The mechanism is called back-propagation, since the errors of each layer depend on that of the next layer so the derivatives are computed from the loss function backwards to the input layers.

### Training Procedure

In summary, the training procedure consists of computing the feedforward process of the neural network, calculating the loss function, back-propagating the gradients, and adjusting the weights. This procedure takes place iteratively for every entry of the dataset, and is called *iterations*. The training process usually exhaust the dataset multiple times in order to better fit the model. Each exhaustion of the dataset is called an *epoch*.

### Convolutional Neural Networks

\begin{figure}[h]
	\includegraphics[width=0.8\textwidth]{./figures/conv.tikz}
	\centering
	\caption[Example of a 2D Convolution function]{Example of a 2D Convolution function, with (a) as input, (b) as the filter, and (c) as the output.} \label{fig:conv}
\end{figure}

A convolutional neural network is a form of neural network which has been proven to be effective in larger input data, such as images. It is similar to fully connected networks seen in Section \ref{neural-networks}, in that it processes the input data using learnable weights and can have activation layers after. However, the weights of a convolutional neural network is organized as a filter, which serves as a window that moves through the input data to generate an output. In Figure \ref{fig:conv}, (b) shows the weights organized as a window for a 2D convolutional neural network. In this figure the filter (b) moves through the input (a) and generates an output, which is the sum of all products between the input values within the window and the weights of the filter. This computation results in calculating the convolution function (or generally, a cross-correlation function), between the input and the filter, hence the name of the network.

This arrangement of the network result in local connectivity of the outputs to the inputs; i.e. the outputs are dependent on only a subset of the input. The extent of this connectivity is called the receptive field, which is the filter size in a single layered convolutional neural networks.

There can be a number of filters that process the inputs; in this case, each filter generates a *channel* in the output. For 2D convolutional neural networks, it may be easy to understand the idea of channels as the "depth" of the data. If the input has multiple channels, each filter generates a single channel using all input channels within the scope of the filter. In other words, a single output value can be thought of as a single perception output, which takes all channel inputs as its input. Consequently, the number of weights in a single convolutional layer equate to the product of the filter size, the number of filters, and the size of the channel dimension of the input.

To control the receptive field and the size of the output, we may add padding on any side of the input, or assign a higher hop length, or *stride*, of the window. Moreover, we can apply the filter sparsely using *dilation*; with a dilation of 2, the filter would take every other input value instead of having a continuous window. Additionally, we may use a pooling layer to control the receptive field without adding additional weights. These layers, alike convolutional layers, have moving filter with optional padding or stride, but instead of aggregating weighted values either selects the maximum or averages all elements within the filter window.

### Residual Neural Networks

\begin{figure}[h]
	\includegraphics[width=0.5\textwidth]{./figures/resnet.png}
	\centering
	\caption{A dataflow representation of a single layer of Residual Neural Network}
\end{figure}

Residual Neural Networks (ResNets) are constructs that utilize convolutional layers with skip connections, where the initial input is added to the output of the convolutional layer. This solves the diminishing gradient problem, where in a deeper neural network the gradient value may become very small during back-propagation  and underflow, which prevents the model from training. By adding the skip connection, the network uses the activations of the input until the weights of the convolutional layer learns to mute the input and return sufficiently significant results. ResNets allow for training deeper neural networks, and they are used frequently for many network models that are reviewed below. [@DBLP:journals/corr/HeZRS15]

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

Such methods of style transfer for audio has been investigated by Ulyanov and Vadim Lebedev [@Ulyanov2016]. Since they lack a multi-class classifier for audio representations that perform as well as VGG19, they used a single convolutional layer to generate the representation needed. While some of their results may sound believable, their results lack theoretical reasoning, contrary to their image style transfer counterpart. Therefore, other mechanisms using Generative Adversarial Networks have been investigated.

## GAN Style Transfers

Isola et al. introduced CycleGAN and `pix2pix`, which performs image style transfer without any pre-trained network. By using generative adversarial networks (GANs), these models learns not only the mapping between the two domain of images but also the loss function and therefore allow a general-purpose solution to image style transfer, without designing a domain-specific loss function. [@DBLP:journals/corr/IsolaZZE16]

### Generative Adversarial Networks

Generative Adversarial Nets are neural network architectures that arrange two neural networks, a generator and a discriminator, as a two player minimax game. The relationship of the generator and the discriminator is usually compared to a criminal and police; the generator aims to generate outputs that fails the discriminator, while the discriminator tries to correctly label the real observation apart from generated data. 

To represent the generator $G$, we define $p_z(\mathbf{z})$ as the input prior distribution, and $G(z; \theta_G)$ as the generative function that represents a multilayer neural network given parameters $\theta_G$. Similarly we define the discriminator function $D(x; \theta_D)$ which outputs a single scalar value that represent the probability of $x$ being a sample from $p_{data}(x)$.

The discriminator is trained to maximize $\log D(x)$ for $x \sim p_{data}(x)$ and minimize $\log D(G(x))$ for $x \sim p_g(x)$, or, in other words, to maximize the probability of assigning the correct source distribution. The generator, on the other hand, is trained to maximize $\log D(G(x))$ for $x \sim p_g(x)$. Given these characteristics, we can define the value function $V(G, D)$ for the minimax game between $G$ and $D$:

\begin{equation}
	V(G, D) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_{z}(z)}[\log (1 - D(G(z)))] \label{eq:gan}
\end{equation}

where $G$ is trained to minimize $V$ and $D$ to maximize $V$.

@NIPS2014_5423 proved that the global optimum of the value function above is when $p_g = p_{data}$, given that the two multilayer neural networks $G$ and $D$ have enough capacity. In practice, however, different rate of training between $G$ and $D$ may cause unstable training. Early in the training, when $G$ fails to generate convincing samples, $D$ can reject samples with high confidence. This causes $\log (1 - D(G(z)))$ to saturate and not provide gradient large enough to train $G$. To mitigate this problem we can train to maximize $\log D(G(z))$ instead. On the other hand, if $G$ trains much faster than $D$, $G$ may collapse too much of $p_g$ to model $p_{data}$. To mitigate this issue, we can update weights for $D$ multiple times for each update of $G$ weights.

We can create a conditional adversarial network that models $p(\mathbf{x} \mid c)$, by adding the condition $c$ as an input to both $G$ and $D$, as we will see in `pix2pix`.

### `pix2pix`

\begin{figure}[h]
	\includegraphics[width=.7\textwidth]{figures/pix2pix.tikz}
	\centering
	\caption[\texttt{pix2pix} model dataflow]{\texttt{pix2pix} model dataflow. Each $\mathcal{L}$ is a loss value calculated by computing the mean squared error between the discriminator output and the correct label. \label{fig:pix2pix}}
\end{figure}

`pix2pix` use conditional adversarial networks to create a general-purpose image style transfer mechanism, given a one-to-one paired dataset. Each pair of data observation has a vector sampled from a domain $X$ (i.e. a photograph), and a range $Y$ (i.e. a painting). We define a generator function $G(\vec(x), \theta_G)$ much alike to the Generative Adversarial Network shown above, but have it model a mapping from $X$ to $Y$. We also define the data distribution of $X$ as $p_{data}(X)$ and that of $Y$ as $p_{data}(Y)$. With the given definition, we modify the GAN value function defined in Equation \ref{eq:gan} to:

\begin{multline}
	V(G, D) = \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{X}), \mathbf{y} \sim p_{data}(\mathbf{Y})}[\log D(x, y)] \\ +  \mathbb{E}_{\mathbf{x} \sim p_{data}(\mathbf{X}), \mathbf{z} \sim p_{z}(\mathbf{z})}[\log (1 - D(x, G(x, z)))] \label{eq:pix2pix}
\end{multline}

The implementation would look like Figure \ref{fig:pix2pix}, where each $\mathcal{L}$ is a loss value calculated by computing the mean squared error between the discriminator output and the correct label.  To use `pix2pix` for music style transfer, however, it is necessary to have a set of paired data, which is hard to acquire. Therefore, I took a closer look into CycleGAN, which does not require a paired dataset.

### CycleGAN

\begin{figure}[h]
	\includegraphics[width=\textwidth]{figures/cyclegan.tikz}
	\centering
	\caption[Dataflow of CycleGAN architecture]{Dataflow of CycleGAN architecture. $G_{A \rightarrow B} = G$ and $G_{B \rightarrow A} = F$ from Equations \ref{eq:cyclegan_1} and \ref{eq:cyclegan_2}. \label{fig:cyclegan}}
\end{figure}

CycleGAN has been introduced to make style transfer possible without a paired dataset. It does so by training a pair of generators and discriminators, arranged as shown on Figure \ref{fig:cyclegan}. We define the following:

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


\paragraph{Constant-Q Transform} Constant-Q Transform is another time-frequency transform which unlike STFT does not use a fixed frequency window, but uses a exponentially spaced frequency bands. Frequency cutoffs are defined as $\omega_k = 2^{\frac{k}{b}} \omega_0$ for $k \in \{ 1, ..., k_{max}\}$, where $\omega_0$ is the starting frequency, $k_{max}$ the number of frequency bands to compute (i.e. the height of the resulting matrix), and $b$ is the scaling factor. Each frequency band $\Delta_k = \omega_k - \omega_{k-1} = \omega_k 2^{\frac{1}{b} - 1}$, which gives a constant frequency to resolution $Q = \frac{\Delta_k}{\omega_k}  = 2^{\frac{1}{b} -1}$. Resulting matrix reflect human perception of sound; when $b$ is set to 12, the frequency cutoffs match the spacing between each semitone in an equally tempered scale. [@DBLP:journals/corr/abs-1811-09620]

\subparagraph{Griffin-Lim} We can recover the original audio from the STFT matrix using Griffin-Lim algorithm. The details of the Griffin-Lim algorithm is beyond the scope of this project; however, we will later use an implementation to recover an audio sample from the neural networks. 

\paragraph{STFT vs CQT} Since CQT uses a logarithmic scale of frequency (due to the exponential cutoffs) and has explicit parameters for starting and ending frequencies, we can ensure that all frequency ranges used by modern music. It also has a higher resolution on lower frequencies, which allows for a better representation of instruments in the lower frequencies. [@DBLP:journals/corr/abs-1811-09620] Given that higher frequencies are less frequent in most songs, CQT seems to be the appropriate representation. 

\paragraph{Raw audio vs Time-Frequency Analyses} Dieleman et al. compared performances of audio classification and auto-tagging tasks between raw audio and intermediate representations. While the performance of the classifiers using intermediate representations work better, they have also shown that neural networks using raw audio can also autonomously learn the features to a comparative degree.[@6854950] WaveNet, which we will cover later in this section, uses raw audio as an input, and learns timbral qualities, such as human voice.[@DBLP:journals/corr/OordDZSVGKSK16]

\paragraph{Mel Frequency Scale} To mitigate the resolution limit in lower frequencies for STFT matrices, we scale the frequencies of the raw audio to a logarithmic scale before computing the STFT matrix. We use the follow equation to do so--given frequency $f$:

\begin{equation}
	m = 2595 \log_{10} (1 + \frac{f}{700})
\end{equation}

STFT matrices that are generated from Mel transformed scales will be referred as Mel-STFT matrices from here on.


### WaveNet 

\begin{figure}[!h]
	\includegraphics[width=0.7\textwidth]{./figures/wavenet.png}
	\centering
	\caption[Diagram of WaveNet]{Diagram of a variant of WaveNet Architecture, implemented by Nvidia} \label{fig:wavenet}
\end{figure}

Figure \ref{fig:wavenet} shows the architecture of WaveNet, introduced by Oord et al., which learns to generate raw audio. [@DBLP:journals/corr/OordDZSVGKSK16] WaveNet generates new sample points that are conditional to the outputs of the previous time step, i.e. the each element of the output waveform of length $T$,  $\mathbf{x} = \{x_0, ..., x_T\}$, is generated by the product of the probability distribution that is conditional to all previous time step:

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

\paragraph{Gated Activation} WaveNet uses two activation function $\tanh$ and sigmoid as shown in Figure \ref{fig:wavenet}. By multiplying the results given by the two activation functions, the sigmoid function acts as gate to limit the effect of the result generated by the $\tanh$ activation. Such ideas have been borrowed from LSTM, a type of recurrent neural networks, and have been shown effective by PixelCNN. [@DBLP:journals/corr/OordKVEGK16] @DBLP:journals/corr/OordDZSVGKSK16 found that using gated activation unit increase the performance of WaveNet significantly.

\paragraph{Conditioning} Alike conditional adversarial networks, we can generate a WaveNet model that is conditional on an external input, i.e. model the $p(x_t | x_0, ..., x_{t-1}, c)$ given a conditional input $c$. Deep Voice uses such conditional model, using voice content as conditional input to generate a Text-to-Speech engine. We will use the conditional model to input pitch data, as we will see in the section below for TimbreTron and Universal Music Translation Networks.

### TimbreTron

\begin{figure}[h]
	\includegraphics[width=\textwidth]{figures/timbretron.tikz}
	\centering
	\caption[TimbreTron model architecture]{TimbreTron model architecture. CQT matrices are notated $CQT$ and the raw audio $PCM$ to distinguish the type of data generated.} \label{fig:timbretron}
\end{figure}

TimbreTron is a combination of CQT, CycleGAN, and WaveNet that has been shown to be effective for timbre transfer. Instead of images, it uses CQT matrix as an input to the generators for CycleGAN in order to transform it to that of a different instrument. The generated matrix is used as a conditional input to the WaveNet synthesizer, which generates a natural audio sample.

\paragraph{Full Spectrum Discriminator} The discriminator used by the original CycleGAN model, PatchGAN, computes the likelihood probability on 70x70 patches of the image and takes the mean of each patch loss. Since such discrimination process do not make sense when dealing with audio data, the discriminator has been modified to compute the entire frequency spectrum. The details of the discriminator design is not included in the paper, so in the implementation chapter I will present different full-spectrum discriminators of original design.

\paragraph{WaveNet} TimbreTron uses a conditional WaveNet to generate the audio from the CQT matrix. Although not specified in the paper, The CycleGAN model and WaveNet are assumed to be trained separately to yield a theoretically sound model. If so, it is questionable whether they needed to use CycleGAN to transform the CQT data to resemble that of the other instrument, as our objective can be simplified to extracting the pitch data from the CQT representation.

\paragraph{Gradient Penalty} TimbreTron uses the GAN value function introduced by the Wasserstein GAN model, along with gradient penalty as the Lipschitz constraint introduced by Gulrajani et al. The resulting value function given posterior distribution $p_g$ from the generator and random sample $p_z$ is:

\begin{equation}
	V(G, D) = \underbrace{
		\mathbb{E}_{x \sim p_{g}(\mathbf{x})} [D(x)] - \mathbb{E}_{x \sim p_{data}(\mathbf{x})} [D(x)]
		}_\textrm{Wasserstein GAN value function} +
		\underbrace{
			\mathbb{E}_{x \sim p_z(\mathbf{x})} [(\| \nabla_x D(x) \|_2 - 1)^2]
		}_\textrm{Gradient Penalty}	
\end{equation}

which has been shown by @DBLP:journals/corr/GulrajaniAADC17.

By using gradient penalty the authors of TimbreTron claim that they were able mitigate the unstable training behaviour shown by the model caused by the fast training speed of the discriminator.

\paragraph{Diminishing Identity Loss} To make sure the generators preserves the pitch content within the matrix during the initial part of the training process, TimbreTron introduces the identity loss (as reviewed in Section \ref{identity-loss}). However, they decided to decrease the weighting of the identity loss throughout the training process since it can hinder the legitimate style transfer process as the training continues.

### Universal Music Translation Network


\begin{figure}[h]
	\includegraphics[width=0.8\textwidth]{figures/facebook_umtn.eps}
    \centering
	\caption[Universal Music Translation Network Diagram]{Universal Music Translation Network Diagram, as shared by Mor et al. \label{fig:facebook_umtn}}
\end{figure}

The Universal Music Translation Network (UMTN) by Mor et al. is another method for timbre transfer which uses the temporal encoder introduced by Engel et al. for Google Magenta project. UMTN extracts the pitch data from the source data using said encoder, and provides it as the conditional input to the WaveNet module. The paper states that such method has been effective for timbre transfer.

\begin{figure}[h]
	\includegraphics[width=\textwidth]{./figures/nsynth.png}.
	\centering
	\caption[NSynth WaveNet AutoEnocder]{NSynth WaveNet AutoEncoder by Engel et al.} \label{fig:nsynth}
\end{figure}

#### The Temporal Encoder

Engel et al. introduced an encoder network design which when combined with a conditional WaveNet decoder acts as an autoencoder, as shown in Figure \ref{fig:nsynth}. The encoder is a chain of convolutional layers with increasing dilation and residual connections much like WaveNet, but with ReLU activations. The paper showed that the resulting vector $Z$ can encode temporal features. For UMTN, we aim to use this Temporal Encoder to extract pitch from the input audio. 

One integral design decision made for UMTN is that the encoder is shared by all inputs to the network model, while they train a WaveNet decoder for each instrument domain. This domain-independent encoder allows timbre transfer for audio input of an instrument not seen during the training process.

#### Pooling and Interpolation

The output of the temporal encoder is downsampled by a pooling layer, and is interpolated again to match the length of the audio sample. This was done so in order to better capture the pitch data, without being altered by short variations which are most likely to be generated by the lesser harmonics that define the timbre of the audio sample.

\begin{figure}[h]
	\includegraphics[width=0.8\textwidth]{figures/dann.eps}
    \centering
	\caption{Structure of Domain Adversarial Neural Networks \label{fig:dann}}
\end{figure}

#### Domain Confusion

To further aid the domain independence of the output of the temporal encoder, the model includes a domain confusion network. It takes the output of the temporal decoder as the input and tries to predict its source domain. By training to maximise the loss function instead, as shown effective by Domain Adversarial Training of Neural Networks by Ganin et al., we train the encoder to output the pitch data regardless of the domain specific features (i.e. timbre). [@Ganin:2016:DTN:2946645.2946704] @DBLP:journals/corr/abs-1805-07848 does not mention how they have implemented their domain confusion but cites @Ganin:2016:DTN:2946645.2946704 instead. Therefore I have assumed here that they have used the domain classifier as shown in Figure \ref{fig:dann}, as that was the original design of @Ganin:2016:DTN:2946645.2946704. Here the sign of the gradient of the classification loss $\mathcal{L}_d$ is flipped during back-propagation instead of training on negative loss in order to prevent the network from learning the sign flip.

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
	[\mathcal{L}_{decoder}(D_i(E(O(x, r), s), t))  - \lambda \mathcal{L}_{dc}(C(E(O(x, r), s), u))] \label{eq:umtn}
\end{multline}

where $\mathcal{L}_{decoder}$, $\mathcal{L}_{dc}$ are Cross Entropy loss functions. The negativity of the domain confusion loss was initially implemented as a gradient reversal layer; however, as will be discussed in Section \ref{loss-function-modification}, the loss function was modified for the actual implementation.


# Implementation

The networks shared above is the current state of art in audio processing using neural networks. In this section I will implement network architectures inspired by these networks. I have used `PyTorch` to implement all models and data processing, and `librosa` to preprocess the audio samples. [@paszke2017automatic; @brian_mcfee_2018_1252297] All hyper-parameters and preprocessing options are passed in through command-line options to allow for more flexible and faster experiment iterations. I have used code written by Zhu et al. [@DBLP:journals/corr/ZhuPIE17; @DBLP:journals/corr/IsolaZZE16] for CycleGAN image style transfer as the boilerplate of the project, but has been heavily modified to accommodate for flexible dataset combinations, diverse neural network models, and distributed model training (i.e. training a single model across multiple GPUs).

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

\paragraph{Free Music Archive} The Free Music Archive has a set of license-free music, classified by genre. Defferrard et al. compiled the songs from the archive into 30 second clips along their metadata information organized into a csv file. [@fma_dataset]

\paragraph{Youtube} With thanks to `MellowBeatSeeker`, a Korean YouTube channel that streams "Chill Lo-Fi Study Beats", I could gather 60 hours of curated chill music. These files were processed using `librosa` to be cut into non-silent intervals.

\paragraph{Maestro} Maestro is a dataset of piano music compiled by the Google Magenta Project. It includes 172 hours worth of MIDI and wav files. They were gathered by using Yamaha Disklavier, which can record MIDI data from acoustic piano keys. Only the audio data has been used for this project. [@hawthorne2018enabling]

\paragraph{GuitarSet} GuitarSet is compiled by Xi et al. at NYU Music and Audio Research Lab using hexaphonic pickups and microphones on acoustic guitars. The dataset contains pairs of soloist and accompanist recordings, which are mixed using `numpy` when the dataset is loaded. [@Xi2018]

### Preprocess 

The preprocessing pipeline processes the sliced data samples into the necessary form needed for different models. Each model sets the default preprocessing options to indicate necessary elements, and the data samples are processed "to spec" before being converted into `PyTorch` tensors. I used `librosa`, a python audio processing package, to convert the data samples into Mel frequency scale and generate STFT or CQT matrices. 

If the model uses STFT or CQT, the output matrix is separated into magnitude and phase by calculating the absolute value and the angle of the complex value. The magnitude is then $\log$ed and linearly scaled to have a maximum of 1 and minimum of -1. This process, as done by GANSynth and DeepVoice, provides a wider variance and prevents gradient explosion from large input values. Furthermore, to remove any volume differences between the samples when not using time-frequency analyses, I have normalized the volume of all audio samples used for the datasets using `ffmpeg-normalize` prior to any training. 

Quantization and pitch shift, which are needed for UMTN and other WaveNet based models, are part of the preprocessing pipeline as well. Pitch shift is done by `librosa`, on a single random-length interval, chosen within the 25% to 75% mark with respect to the length of the data sample. The amount to be shifted was sampled from uniform distribution between 0 and 1. $\mu$-law function needed for quantization was implemented based on Equation \ref{eq:mulaw}.

#### Postprocess 

The parameters that are needed to recover the original matrix are given to the network model with the data samples to be passed on to the postprocess pipeline. Inverses of all preprocess functions are implemented. STFT and CQT matrices are reconstructed using the Griffin-Lim algorithm which has been implemented by `librosa`. Its implementation of the restoration algorithm for CQT matrices, however, is unstable and the authors of the library do not recommend using it for purposes other than that diagnostic applications. With a simple experiment of applying CQT and restoring it, with $b = 120$ and $k_max = 84$, I could verify that the algorithm can successfully restore the audio for the purpose of this project.

## Network Models

In this section I introduce two neural network models inspired by TimbreTron and UMTN, along with an implementation of UMTN. TimbreTron could not be implemented because the network model was too large to fit into my system; moreover, @DBLP:journals/corr/abs-1811-09620 does not include many details (such as the generator architecture) which is essential to the implementation of the network.

First of the two new models is the CycleGAN model that uses the Griffin-Lim algorithm as the reconstruction mechanism. I tried to implement different generator and classifier networks to use with the time-frequency representations. Unlike generator networks, classifier networks can be evaluated independently in order to test the capacity of the network; hence I have written a separate model to evaluates these networks.

The second model was inspired by UMTN; while the UMTN learns pitch data from the temporal classifiers, I wondered if the results could change by using an encoder network with a time-frequency representation. The encoder networks from the CycleGAN model were used to implement such a model.

### Classifiers

\begin{figure}
	\includegraphics[width=\textwidth]{figures/classifiers.tikz}
	\centering
	\caption{Visualization of the Classifiers}
\end{figure}

I have experimented with three different classifier networks, one from the original CycleGAN model, and two of my original design. In the paragraphs below, I will explain the details of the networks and the intuition behind them. All classifiers were optimized using the Adam optimizer and the learning rate was adjusted by model to prevent divergence, as will be presented in the Evaluation chapter.

#### `shallow` classifier (PatchGAN)

PatchGAN is the classifier from the original CycleGAN model. It classifies the generated image based on overlapping patches of the image, which generates a prediction on how believable each patch is, instead of returning a single number as the output for the entire image. The patch can be thought of as a large convolutional filter that is implemented by using a number of convolutional layers. The `shallow` classifier network is a reconstruction of the PatchGAN classifier, except that the normalization layers from the original model have been omitted since we are not dealing with image data.

The network consists of five convolutional layers. First four layers have filters with kernel size of 4, padding of 1 on each side, and stride of 2, which doubles the size of the receptive area per layer. The last layer contains of the same kernel size and padding but with no strides. The layers combined create a "patch" with a patch of size 64-by-64, which convolves through the input image to generate a prediction for each patch. The number of filters is doubled for the first four layers, and is downsampled into a single filter on the final one.

#### `conv1d`

@Ulyanov2016 uses the frequency dimension of a time-frequency analyses as the filter input to convolutional neural networks, and claims that such methods have been effective for the non-GAN style transfer they implemented. Since there is a filter for each input and output channel combination, by doing so we effectively generate a fully connected network of filters, which takes the frequencies data as the input. This classifier network was created largely to verify whether it is truly effective to use frequency dimension as the channel input to convolutional networks.

A number of parameters need to be defined to explain this network: `ndf` is the number of filters (i.e. the size of the channel dimension of the first layer), `mdf` the multiplier between the layers of the network, and `n_layers` is the number of convolutional layers the network contains.

The network includes a series of 1D Convolutional layers; the first layer upsamples the input into `ndf` channels, and each layer, up until the penultimate one, has $\texttt{mdf} * c_n$ channels, where $c_n$ is size of the channel dimension of the previous layer. The last layer downsamples the channel input down to one. Given $\texttt{n\_layers} = 4$, and a kernel size of 3, this network has a receptive field of 31.

Multiple values for each parameters were used for training. Namely, 1024 and 2048 were values chosen for `ndf`, 1 and 2 for `mdf`, and 2, 3, and 4 for `n_layers`. This network, however, failed to converge during training with any combination of these values.

#### Timbral classifier

Lastly, the timbral classifier network extends the `shallow` model to include the entire frequency spectrum inside the receptive area. Depending on the size of the input, it creates layers with filter size of 3 or 4, depending on whether the input size is odd or even. All layers have a stride of 2 on frequency dimension and 1 (no stride) on time dimension. Like the `conv1d` classifier, it generates `ngf` channels on the first layer, and downsamples it down to a single channel on the last layer. This generates a large patch that covers the entire frequency spectrum and has a variable patch size on the time domain. For example, if the given input has a size of 880 on the frequency dimension, it will result in a total receptive length of 23 on the time dimension.

Authors of Timbretron claimed that `shallow` classifier did not work for time-frequency representations because of its locality, and argued for full-spectrum discrimnators. I have further hypothesized that the `shallow` classifier would not perform well for polytimbral audio samples. When different instruments play simultaneously, they cover different ranges of the frequency spectrum. In such cases, patches generated by the `shallow` network would not be able to discriminate between higher and lower frequency bands and fail to classify. For these reasons, the timbral classifier was designed to discriminate based on the full frequency spectrum of the input.

### CycleGAN

Timbretron introduces a model that uses Griffin-Lim on STFT matrices as the baseline, but fails to present the equivalent for CQT representations. By implementing a CycleGAN model that uses Griffin-Lim reconstruction on both STFT and CQT representations, I hoped to clarify (1) the necessity of WaveNet in the TimbreTron model, and (2) the differences between STFT and CQT CycleGAN models. Classifier networks designed and evaluated above are used as discriminators of the network architecture, and the encoder network is a variation of the original encoder model.

While many encoder models, such as those that use recurrent neural networks (RNN), were implemented but showed quick divergence of the model due to gradient explosion, a common training difficulty for RNNs. The details of RNNs lay outside the scope of this project so will not be dealt any further.

The model was trained with a learning rate of 0.0002, with the diminishing identity loss used for TimbreTron. The original codebase by Zhu et al. had an implementation of Wasserstein GAN with Gradient Penalty, but it failed to perform reliably. Least Squared GAN, which is used by the original CycleGAN paper, was used instead. [@DBLP:journals/corr/ZhuPIE17]
 
#### The Encoder

The encoder consists of three parts: downsampling, transformation, and upsampling. The downsampling layers where implemented using the same techniques that was used for the timbral classifier network. The filter height was set to 4 if the input height was even and 3 if it was odd, and identical measures were implemented for the filter width as well. After three layers of downsampling, which decreases each dimension to eighth of its size, the input tensors are transformed by a series of residual network layers. I have chosen to use 9 transformation layers for the encoder, which is the value the original CycleGAN model found effective for image transfer. Any higher number of transformation layers made the model difficult to run on a single GPU. Lastly, same number of transposed convolutional layers were added for the purpose of upsampling the output of the transformation layers into the original size of the tensor. All layers in the network, except the last layer, use ReLU as the activation function. 

#### Distributed Training System

\begin{figure}[h]
	\includegraphics[width=\textwidth]{./figures/cyclegan_gpu.tikz}
	\centering
	\caption[Distributed CycleGAN Diagram]{A modification of Figure \ref{fig:cyclegan} to show how the model is split between the two GPUs. The two yellow points show the point at which the tensor has to be transferred; this way we can minimize the necessity of transferring the output vector.} \label{fig:cyclegan_gpu}
\end{figure}

Some generator models made the entire network architecture too large to fit in the memory of a single GPU. The original model implemented by Zhu et al., however, only had multi-GPU training capacities using dataset distribution. Therefore, the code structure was modified to store a generator and a discriminator pair on each GPU, and the relevant device management code was modified. With distributed modelling techniques, the transfer of tensors between the GPU can become a major bottleneck within the system; therefore, the modified codebase relies on individual model implementation to design ways to transfer minimal information between the devices. For the CycleGAN module, this was done as seen in Figure \ref{fig:cyclegan_gpu}, which needs only two transfer between the devices for a single iteration. `PyTorch` handles the transfer of gradients between devices during back-propagation so there was no worry of losing gradient information.

### TimbreTron

With the distributed training in place, I was hoping to be able to train TimbreTron on this system; however, there was still not enough memory resources to train the six networks together. Without looking further ways to solve this issue, I have decided to implement the Universal Music Translation Network which included some promising audio samples within the paper.

### Universal Music Translation Network

\begin{figure}[h]
	\includegraphics[width=\textwidth]{figures/umtn.tikz}
\centering 
\caption{Implemented architecture of the Universal Music Translation Network \label{fig:umtn}}
\end{figure}

The penultimate model of this project on is a recreation of the Universal Music Translation Network. At the time of the implementation of the project there was no code implementation that verified the efficacy of the network architecture. Therefore, the implementation presented by this project is based on the details of the network written in the paper.

The temporal encoder is a faithful recreation of the one presented in Figure \ref{fig:nsynth}, although the width (the channel dimension) of the encoder was left as a hyper-parameter. The model initially introduced by the authors were trained on 6 8-GPU machines for 6 days, which was not a viable option for this project. Therefore while the original paper uses a channel width a 128 for both the encoder and the WaveNet decoder, for faster training this project uses a width of 64 for both networks. The width of the final layer of the encoder was kept as 64 in accordance with the paper. The results of the encoder was pooled with a filter and stride size of 512, instead of 400 which was used by the original model, because the sample rate of the audio samples was increased from 8000 to 8192Hz.

Moreover, the details of the domain confusion network are absent from the referenced paper, other than the fact that it contains three convolutional layers and computes the mean of the output. Through trial-and-error, therefore, filter size of 1 and channel size of 64 were selected.

\paragraph{\texttt{nv-wavenet}} A modified version of WaveNet implemented by Nvidia was used for the decoder for faster inference. Nvidia's implementation includes an optimized WaveNet inference kernel for their GPUs, which drastically increases the speed of inference. The upsampling layer in the implementation was removed, since the architecture includes a separate interpolation layer which extends the encoder input to the length of the audio. Related elements for conditional input processing were modified in coherence with this change.

\paragraph{Loss function modification \label{loss-function-modification}} The domain confusion loss either quickly diverged or diminished to zero when trained together with the WaveNet reconstruction loss. This was due to the fact that there was no procedure to train the classifier network to classify correctly. Therefore the original loss function for the UMTN was divided into two different optimization processes. First is the reconstruction loss function, which is equal to the one introduced in Section \ref{universal-music-translation-network}:

\begin{multline}
	\mathcal{L}_{recon} = \sum_{i \in I} \mathbb{E}_{x \sim p_{data}(x), \{r, s, t, u\} \sim p_{z}} \\
	[\mathcal{L}_{decoder}(D_i(E(O(x, r), s), t))  - \lambda \mathcal{L}_{dc}(C(E(O(x, r), s), u))] \tag{\ref{eq:umtn}}
\end{multline}

which was used to optimize the encoder $E$ and decoders $D_i$. Secondly, the negative clause of this loss function was calculated separately to optimize the classifier $C$, i.e. the equation

\begin{equation}
	\mathbb{E}_{x \sim p_{data}(x), \{r, s, t\} \sim p_{z}} [\lambda \mathcal{L}_{dc}(C(E(O(x, r), s), t))] \label{eq:dcloss}
\end{equation}

was trained to optimize $C$ for classification. This procedure stabilized the training behaviour, and resulted in the desired output.

Moreover, the paper states that they have trained the model with a learning rate of 0.001, as per the default value of the Adam optimizer. However, the model quickly diverged with the said learning rate even with exact parameter values including a width of 128. With a smaller learning rate of 0.0001 and the new domain confusion loss calculation mechanism, the model showed convergence.

\paragraph{Limitations \label{limitations}} The authors of UMTN trained the network for 6 days using a lot more hardware resources. To achieve a similar scale of computation, this model would need to train for more than a month on the hardware that I had access to. Therefore, samples were taken from epochs before full convergence, which limited the quality of the audio samples used for evaluation.

### Spectral Translation Model

\begin{figure}
	\includegraphics{figures/spectral.tikz}
	\centering
	\caption{Diagram of the Spectral Translation Model}
\end{figure}

I have designed another model which borrows the same domain-adversarial network design from UMTN, but uses the generator and discriminators designed for CycleGAN models. Deep Voice and earlier conditional WaveNet models used Mel-STFT inputs to WaveNet to create voice transfer. [@DBLP:journals/corr/ArikCCDGKLMRSS17] Similarly, here I use encoder and classifier networks designed for such spectral representation in order to condition the WaveNet decoder. Unlike UMTN, there is no pooling layer for this model; however the output of the encoder is interpolated by the nearest neighbor algorithm like the original model.

### Evaluation Preparation

The results generated by the models can only be tested by human ears, as styles of sound are defined by human perception. Therefore, this project includes a small web service that is meant to serve the questionnaire and aggregate the results. It is implemented using `flask`, `sqlalchemy`, and `wtforms`, each used to serve web requests, handle database updates, and form validation. The web sites shows a simple `bootstrap` website, which guides the participant through the consent form, instructions, and the question set. All responses were saved on a `sqlite` database; there was no need for a fully functional SQL database since the database was only accessed by a limited number of people executing simple operations.

# Evaluation 

All models implemented were trained with two sets of datasets. The first was a pairing of the Maestro piano dataset and GuitarSet, and the second was a pairing between samples of Pop and Rock from the FMA dataset, paired with the curated playlist by MellowBeatSeeker. The objective of these dataset pairing was to define two different domains of different musical style. The first pair made a clear distinction in domain by timbre. The second pair, on the other hand, was an attempt to capture a more subtle musical style of "chillness", for which people do make a clear distinction, evidenced by the number of curated "chill" playlists available. Since it is not possible to acquire a curated play list of music not considered chill, a set of music from related genres are selected instead. Admittedly, the two domains are not mutually exclusive in this case, but in majority of the samples there exists a distinction that allows classification. The first dataset from here on will be referred as the `piano2guitar` dataset, since it will be used for style transfer between piano and guitar, and the latter as the `chillify` dataset, since it will be used to train networks that can "chillify" a piece of music.

## Classifier Models

\begin{table}[h]
\begin{tabular}{cc|c|c|c|c|}
\cline{3-6}
                                                &                & \multicolumn{2}{c|}{piano2guitar} & \multicolumn{2}{c|}{chillify} \\ \cline{3-6} 
                                                &                & linear  & sigmoid & linear & sigmoid \\ \hline
\multicolumn{1}{|c|}{\multirow{2}{*}{Mel-STFT}} & ndf:4, mdf:2   & 1.0     & 0.99    & 0.67   & 0.70    \\ \cline{2-6} 
\multicolumn{1}{|c|}{}                          & ndf:64, mdf: 1 & 0.99    & 1.0     & 0.69   & 0.65    \\ \hline
\multicolumn{1}{|c|}{\multirow{2}{*}{CQT}}      & ndf:4, mdf:2   & 1.0     & 1.0     & 0.695   & 0.66    \\ \cline{2-6} 
\multicolumn{1}{|c|}{}                          & ndf:64, mdf:1  & 1.0     & 0.99    & 0.62   & 0.775    \\ \hline
\end{tabular}
\centering
\caption{The accuracy results of the \texttt{timbral} classifier with different parameters} \label{tab:timbral}
\end{table}

\begin{table}[h]
\begin{tabular}{llll}
\hline
Representation            & ndf & piano2guitar & chillify \\ \hline
\multirow{2}{*}{Mel-STFT} & 32  & 1.0          & 0.67     \\
                          & 64  & 0.99         & 0.765     \\
\multirow{2}{*}{CQT}      & 32  & 1.0          & 0.925     \\
                          & 64  & 1.0          & 0.61     \\ \hline
\end{tabular}
\centering
\caption{The accuracy results of the \texttt{shallow} classifier with different parameters} \label{tab:shallow}
\end{table}

The first set of experiments was the evaluation of a number of binary classifiers, introduced in Section \ref{classifiers}, against the two pairs of datasets. A number of classifiers were trained with different parameters in order to select the best classifier to be used as the discriminator for the network models below. Since the `conv1d` classifier did not converge, only the `timbral` and `shallow` classifiers were trained until convergence. For the `timbral` classifier, networks with a initial filter number (`ndf`) of 4 and filter multiplier (`mdf`) of 2, and networks with constant filter number of 64 were trained, both with and without a sigmoid activation on the final layer of the network. All `shallow` classifier, on the other hand, were trained without the sigmoid activation to adhere to the PatchGAN classifier architecture, but multiple number of filters, 32 and 64, were used. All classifiers are trained on both Mel-STFT and CQT representations.

Since the dataset is balanced, comparing the accuracies of the classifiers is sufficient to evaluate the performance of the network. The accuracy results are shown by tables \ref{tab:timbral} and \ref{tab:shallow}. These results were computed on a separate validation dataset, in order to choose the best set of parameters before comparing the results between the two classifiers on the test dataset. However, the results from the `piano2guitar` dataset show near perfect accuracy on the validation dataset for all models, which made it unfeasible to try to choose the best classifier based upon these results. Results varied for the `chillify` dataset, however, and therefore was used to evaluate the classifiers. Additionally, the results do not show a favour towards any hyper-parameter value.

\begin{table}[]
\begin{tabular}{cc}
\hline
Classifier & Accuracy \\ \hline
timbral    & 0.705     \\
shallow    & 0.935   \\ \hline
\end{tabular}
\centering
\caption{Results from the testing dataset, comparing the two classifiers} \label{tab:classifier}
\end{table}

From the results of the validation set, the `timbral` classifier with a constant filter size of 64 and the sigmoid activation on the output on CQT representations and the `shallow` classifier with the channel size of 32, equally using the CQT representations were selected for evaluation on the testing dataset. Table \ref{tab:classifier} shows the accuracy results of the two networks. Contrary to the results shown by TimbreTron, the `shallow` translator performed better than the `timbral` classifier by a significant margin, refuting my hypothesis on the effectiveness of a full-spectrum discriminator. It is especially noteworthy since the size of the `shallow` classifier is significant smaller than the `timbral` classifier. To see if the results can defeat the null hypothesis, a significance test was performed, which resulted in a $p$ value of $0.0014 < 0.02$ which is significant.

## CycleGAN models

Using the `shallow` classifier selected from above, a number of CycleGAN model was trained on both `piano2guitar` and `chillify` datasets but failed to successfully transfer sound qualities between the two domains. The model was trained with different encoder parameters. The number of downsampling convolutional layers (`n_downsample`) ranged from 2 to 4, the number of ResNet layers (`num_trans_layer`) from 6 to 10, the number of filters (`ngf`) in the generator were selected between 16, 32, and 64. During training of some of these networks, the discriminator converged a lot earlier than the generator, preventing the generator from training. The weight for the discriminator loss was adjusted accordingly for each experiment. All combinations of these parameters were experimented, but did not resulted in the desired domain transfer. Some artifacts of domain transfer appear earlier in the training process for some experiments, but as the model converged the said qualities disappeared. 

Some of the effects that the CycleGAN model showed instead, included:

1. Muting, which happened if the discriminator converged faster than the generator,
2. No transformation, which happened when the identity loss was set too high earlier, or if the generator converged much faster than the discriminator,
3. Vocal Removal, which happened to models that were trained with the `chillify` dataset. Since a lot of the samples in the `fma` dataset included vocal tracks and none did in the `youtube` dataset, the network learned to mute the vocals from the audio samples instead. This was not foreseen when compiling the dataset and is not an effect I was hoping to see, but is a valid domain transfer.
4. Volume Attenuation, which also happened to models that were trained with the `chillify` dataset. This was mitigated by linear normalization of all spectral representations.

Most training models showed effect 1 or 2, and even with weight adjustment did not show a particularly different behaviour. 

## Universal Music Translation Network

The Universal Music Translation Network, on the other hand, showed promising results after the training process was modified, as explained in Section \ref{loss-function-modification}. It, as described in the original paper, trained very slowly; it took 14 days of training to get a reasonably sounding sample used for the human perception experiment. When trained with the `piano2guitar` dataset, the WaveNet decoder recognized timbre of the instruments fairly earlier on during the training process (around 3 days into training), but for the pitch data to be correctly extracted and condition the decoder it took much longer.

While the model loss converged when trained with `piano2guitar` dataset, it diverged with the `chillify` dataset, even with different values of $\lambda_{dc}$ to incentivize stronger classification and pitch recognition. This was expected, however. The temporal encoder, as shown by @DBLP:journals/corr/EngelRRDESN17, could recognize easily defined audio style such as note qualities and timbre, but could not recognize audio features that humans find harder to define, such as genre. With such an input, the classifier would not have been able to classify the two domains apart due to lack of capacity of the network. This is evidenced by the quick diverging behaviour of $\mathcal{L}_{dc}$ as shown in Figure \ref{fig:umtn_loss}. Especially with the pooling and interpolation layer, which encourages the encoder to attain the collection of fundamental frequencies, i.e. pitch. If the encoder was successful in extracting pitch from the `chillify` dataset pairs, there would have been too much variance in pitch in each domain for WaveNet to correctly model.

\begin{figure}[]
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\textwidth]{./figures/p2g_umtn.png}  
  \caption{Loss function plot for \texttt{piano2guitar} dataset}
\end{subfigure}
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\textwidth]{./figures/chillify_umtn.png}  
  \caption{Loss function plot for \texttt{chillify} dataset}
\end{subfigure}
\caption{Difference is convergence/divergence behaviour between datasets}
\label{fig:umtn_loss}
\centering
\end{figure}

\begin{figure}
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\textwidth]{./figures/sec_1_content.png}  
  \caption{Response for content preservation for timbre transfer from Piano to Guitar}
\end{subfigure}
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\textwidth]{./figures/sec_1_style.png}  
  \caption{Response for style change for timbre transfer from Piano to Guitar}
\end{subfigure}
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\textwidth]{./figures/sec_2_content.png}  
  \caption{Response for content preservation for timbre transfer from Guitar to Piano}
\end{subfigure}
\begin{subfigure}{.5\textwidth}
  \centering
  \includegraphics[width=\textwidth]{./figures/sec_2_style.png}  
\caption{Response for style change for timbre transfer from Guitar to Piano} \label{fig:guitarstyle}
\end{subfigure}
\caption{Aggregation of responses of 39 people to the questionnaire} \label{fig:responses}
\centering
\end{figure}

### Human Perception Experiment

Evaluation with human participants is essential in evaluating whether the universal music translation network successfully translates audio samples from one style to another, since a musical style is itself defined by the consensus of human perception. A human perception experiment was devised, therefore, to ask a number of participants on (1) the similarity in content, and (2) the likelihood of the timbre of the instrument. The questionnaire contained 16 samples organised into two sections; first containing samples generated from piano recordings translated by the guitar WaveNet decoder, and second containing samples generated from the piano WaveNet decoder from guitar recordings. I asked the two questions above for each audio sample, and the answers to the questionnaires were recorded on a five-point likert scale, each labelled: Very Different, Different, Neutral, Similar, and Very Similar, which each corresponds to 1 through 5 in Figure \ref{fig:responses}.


\paragraph{Frequency difference} The answers to the questions shown as histograms is shown by Figure \ref{fig:responses}. Answer to the question of content similarity between the original sample and the generated sample does not have a consensus for both direction of transfer. For section 1, 125 responses noted that the contents did not sound similar while 111 did; for section 2, 121 response noted that the contents were dissimilar, while 114 noted that they were.

\begin{figure}[!h]
	\includegraphics[width=.7\textwidth]{./figures/decoder.png}
	\centering
	\caption{Loss function behaviour of the two decoders over time} \label{fig:decoder}
\end{figure}

For efficacy of style transfer the responses had a consensus. Especially for the timbre transfer from Guitar to Piano, shown by Figure \ref{fig:guitarstyle}, there is a clear consensus. Its mean value of 3.4 notes that the style transfer was successful. This is not the case for the transfer from piano to guitar; with a mean value of 2.7, more people thought that the decoder does not produce audio samples of a quality good enough to generate convincing pieces. However, as mention in Section \ref{limitations} and seen on Figure \ref{fig:decoder}, the decoder has not converged fully, due to hardware and time limitations. The difference in responses is coherent with the loss function behaviour, shown by Figure \ref{fig:decoder}, which shows a better convergence for the piano decoder.

To further investigate the differences in diverse responses in content similarity in both sections, a number of audio samples were investigated to find the pitfalls of the translation network. The questions that ranked the lowest mean value was reviewed for similarities. 

The generated guitar samples that performed badly in the questionnaire had clear similarity; major sections of the input piano samples lied on pitches that were not available on a guitar. The guitar decoder had never seen those pitches during training, and therefore muted those inputs given by the encoder. The resulting sample only included parts of the pitch data sporadically.

Moreover, the generated piano samples that poorly performed in the questionnaire also had a clear similarity. The guitar samples used for generating these samples used finger-style techniques that plays a number of pitches sequentially with very little time period in between the two. Because of the pool length of 512 applied after the encoder, the fastest movement the encoder can capture is 1/16th of a second. Moreover, these type of fast arpeggio is rare and sometimes impossible on pianos, because of the time the hammer takes to get back to its original position. Therefore, the decoder has very little information in both content and style to recreate a realistic sound sample.

## Spectral Translation Model

\begin{figure}[h]
	\includegraphics[width=.7\textwidth]{figures/spectral_loss.png}
	\centering
	\caption{Spectral Translation Model Loss Behaviour}
\end{figure}

The spectral translator model was designed to evaluate whether the ill training behaviour of the CycleGAN model was due to the capacity of the generator network, or the lack of stability in GAN training. Since the generator network in the spectral translator model was trained to remove domain-specific characteristic, which is a simpler modification to the input data than style transfer, the encoder network should have been able to achieve this if the network had enough capacity to achieve style transfer. However when trained with same set of generators and discriminators used for the CycleGAN model on the `piano2guitar` dataset, there was very little variation in the classification loss, and the decoder behaviour after training resembled that of an unconditioned WaveNet response. Since we have shown the effectiveness of the classifier network, this strongly alludes to the lack of capacity in the design of the generator network.

# Conclusion  

## Summary of Works

During this project, a number of neural network constructs, from Generative Adversarial Networks to WaveNet, Timbretron, and the Universal Music Translation Network were reviewed in order to create a model that can transform audio samples by recognizing audio feature.

To approach such goal in a more methodological way, Multiple classifier models that can recognize the audio features mentioned above were designed and implemented based upon previous works, and evaluated their performance with thorough testing of parameters and relative significance testing.

With the successful implementation of a neural network model that can classify audio features, I have used techniques and methods from the reviewed previous works, such as CycleGAN network architecture, to either reimplement an existing model, or design a new network structure that modify the style of input audios signals. Each model was designed to mitigate a set of limitations of the project, such as hardware constraints or the quality of the dataset. Moreover, each model was designed either to verify a claim that a previous work has made, or overcome the lack of information given in the descriptions of those works.

Furthermore, in implementing a successful neural network model, I have conducted a human perception experiment to evaluate the efficacy of the model, and analysed the results to examine the performance and limitations of the neural network model. For models that failed to output the desired output, I have analysed the behaviour of the model and the reason for failure.

## Future Works

As shown by Section \ref{spectral-translation-model-1}, the most likely reason for the failure of the CycleGAN and Spectral Translation Model is the generator model. With a better designed generator model that effectively transform audio features in a time-frequency analysis, the network architecture proposed in this project could work much more effectively and produce usable results. Therefore, a research on transformer networks for time-frequency representations would allow for more effective audio processing using neural networks, although it was not within the scope of this project.

Generally, more research on the area of audio processing using machine learning can allow for better tools for musicians to express their creativity. A software instrument in the current music production process require a lot of storage space with a limited expressibility in sound, because it depends on pre-recorded pieces of samples. With neural networks that learn audio features, a more efficient and effective sound generation that encapsulates subtle audio qualities can take place. Timbre could itself be an area for unsupervised learning algorithms, which would allow for compositions and modifications of instrument qualities that is not possible with traditional audio mixing.

During the research for previous works for this project, I noticed a lack of interest for sound in the machine learning industry, and found that most audio processing techniques using neural networks were confined to that of human voice. While this is an important area of research, especially for more natural human computer interaction, this suggested that researches that may allow us to use the computational methodologies that we have to better understand properties of sound and music were getting ignored. By conducting this project, therefore, I hoped to shine some light to many possible areas of computer science research not only in audio, but in music.

\chapter*{Bibliography}

::: {#refs}
:::

\appendix


# Consent Form

Experiment Purpose & Procedure

The purpose of this experiment is to use neural networks to translate musical styles from instruments or curated playlists. The experiment consists of 2 sections, each with 8 questions, during which you will be asked to listen to a pair of audio samples and answer two questions, one regarding the similarity in content of the two samples, and the other regarding the difference in audio style between the two samples. Each sample will be 4 seconds long, and you can play each sample as many times as you require. The questionnaire includes more detailed instructions for you to follow.

Please note that none of the tasks is a test of your personal intelligence or ability. The objective is to test the efficacy of the neural networks in translating the style of an audio sample to the other.

You may also withdraw from the experiment at any time, and hold no obligation to finish the questionnaire.

## Confidentiality
The following data will be recorded to remove possibilities of duplication and participation gifts, and will be discarded after the experiments conclude:

* Name
* Email address

All data will be coded so that your anonymity will be protected in any research papers and presentations that result from this work.

## Finding out about result

If interested, you can find out the result of the study by contacting the researcher, Charles Yoon, after 17 May, 2019. His email address is jyy24`at`cam.ac.uk.

## Record of Consent

By filling out the form below indicates that you have understood the information about this experiment and consent to your participation. The participation is voluntary and you may refuse to answer certain questions on the questionnaire and withdraw from the study at any time with no penalty. This does not waive your legal rights. You should have received a copy of the consent form for your own record. If you have further questions related to this research, please contact the researcher.

# Questionnaire 


## Section 1
In this section, you will be presented with 8 pairs of sound clips, that are either original or generated from a neural network. This neural network was trained to translate the piano sound to that of the guitar. Please answer the following questions after listening to each pair.

* How similar does the clip sound to a guitar?
* How similar do the two clips sound?

For the first question you will be given two choices, and you can choose the one that sounds more chilled to your taste. For the latter, you will be given a Likert scale, from 'Very Different' to 'Very Simliar'; please choose the option that describes the two clips best. You can listen to both clips as many times as you require to make your choice.

*For every real and generated sample pairs*

*Audio Samples A and B were displayed here for playback*

1) How similar does Sample B sound to a guitar?

*A Likert scale from Very Different to Very Simliar was shown here*

2) How similar does Sample B sound to Sample A?

*A Likert scale from Very Different to Very Simliar was shown here*

## Section 2

In this section, you will again be presented with 8 pairs of sound clips, that are either original or generated from a neural network. This neural network was trained to translate guitar sounds to those of piano. Please answer the following questions after listening to each pair.

* How similar does the sound clip sound to a piano?
* How similar do the two clips sound?

For the first question you will be given two choices, and you can choose the one that sounds more chilled to your taste. For the latter, you will be given a Likert scale, from 'Very Different' to 'Very Simliar'; please choose the option that describes the two clips best. You can listen to both clips as many times as you require to make your choice.

*For every real and generated sample pairs*

*Audio Samples A and B were displayed here for playback*

1) How similar does Sample B sound to a piano?

*A Likert scale from Very Different to Very Simliar was shown here*

2) How similar does Sample B sound to Sample A?

*A Likert scale from Very Different to Very Simliar was shown here*


# Histogram of Answers of each Questions

\begin{figure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_1_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_2_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_3_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_4_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_5_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_6_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_7_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_8_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_1_style.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_2_style.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_3_style.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_4_style.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_5_style.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_6_style.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_7_style.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_1_8_style.png}  
\end{subfigure}
\end{figure}
\begin{figure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_1_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_2_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_3_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_4_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_5_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_6_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_7_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_8_content.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_1_style.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_2_style.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_3_style.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_4_style.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_5_style.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_6_style.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_7_style.png}  
\end{subfigure}
\begin{subfigure}{.24\textwidth}
\centering
\includegraphics[width=\textwidth]{./figures/res_2_8_style.png}  
\end{subfigure}
\end{figure}

# Project Proposal

\begin{center}
\Large
Computer Science Tripos -- Part II -- Project Proposal\\[4mm]
\LARGE
Music Style Transfer\\[4mm]

\large
Candidate Number: 303488330 

Originator: Prof. Alan Blackwell

19 October 2018
\end{center}

\vspace{5mm}

\textbf{Project Supervisor:} Dr. Andrea Franceschini

\textbf{Director of Studies:} Dr. Timothy Griffin

\textbf{Project Overseers:} Prof. Lawrence Paulson  \& Prof. Frank Stajano

## Introduction and Description of Work

This project aims to create a system that modifies a given song to mimic the style of another. Specifically, this project will create a system that achieves such goal by one or both of the following methods:

1. Creating a deep neural network that learns the similarity between songs in a playlist of a common stylistic trait. Such style transfer was done for images by Gatys, Ecker, and Bethge, in their paper "A Neural Algorithm of Artistic Style", in which the network was trained to transfer the style of "Starry Night" by Van Gogh to existing paintings.[^1]

2. Training a generative adversarial network[^2] that learns the transformation of stylistic trait given a set of pairs of song, the pair being similar in content but different in style. This method has been done for images by Isola, Zhu, Zhou, and Efros, which given a base image yields a more detailed image (e.g. from boxes to images of buildings).[^3]

In both cases, styles of music can be perceived differently depending on the individual; therefore, the original dataset will be evaluated to see if the set of songs indeed (1) is a playlist with a common stylistic trait in the case of the first method, or (2) are pairs with the same content (i.e. the "same song"), in the case of the second. Moreover, in the case of second method, similarity between the audio pairs can be analysed by existing mechanisms, that may include the open source library Musly[^4] and the mechanisms described by McFee[^5] and Zadel[^6]. In popular culture this may be referred to as a "remix" or a "cover"; however, this project purports to yield any qualitative change, which may not be as drastic as a remix.

Facebook AI Research team has done a research on transferring musical style previously, and this project largely aims to yield a similar result to their research.[^7] This research, however, aimed to transfer a piece of classical music into another instrument; we may aim to target a more general characteristic of music people perceive, and a different genre of songs in this project. Nonetheless, their paper elaborates their evaluation method in detail and this project will consult on their approach to determine the evaluation method of its own.

[^1]: \tt{arXiv:1508.06576 [cs.CV]}
[^2]: \tt{arXiv:1406.2661 [stat.ML]}
[^3]: \tt{arXiv:1611.07004 [cs.CV]}
[^4]: http://www.musly.org/
[^5]: https://escholarship.org/uc/item/8s90q67r
[^6]: http://www.music.mcgill.ca/~zadel/611/audiosimilarity/similaritypresentation.pdf 
[^7]: \tt{arXiv:1805.07848 [cs.SD]}
[^8]: https://www.deeplearningbook.org/

## Resource Required

### Computing Resource

Since this project contains multiple machine learning experiments, a machine with nVidia CUDA capable GPU would be ideal. The Rainbow group has a machine with a nVidia Titan Xp, which this project will mostly train its network on. As a backup, a personal desktop computer with a nVidia GTX1080 will be used to train networks as well. 

### Dataset

A dataset for this project does not seem to be publicly available; therefore, preparing dataset for this project will most likely involve compiling a set of songs. For the first method, it would most likely involve compiling a playlist with a similar stylistic trait. Since there are a number of "chill" playlists publicly available on Spotify, this project will use such playlist. For the second method, acquiring an entire discography of an cover artist such as Boyce Avenue and the original songs for the listed tracks should suffice.

## Starting Point

Neural Networks were covered by the course "Artificial Intelligence" during Part IB; however, most in depth contents such as Convolutional Neural Network and Generative Adversarial Networks will need to be reviewed and learned as the project starts.

## Substance and Structure of the Project

### Previous Work Research

The field of Machine Learning and structures of Neural Networks is rapidly growing. I will need to review the relevant papers and textbooks for an in-depth understanding of different network architectures and systems, and the mathematics behind them. During this time I will also follow the works done for Image and Sound Style Transfer to grab a better sense of neural networks.

### Data Preparation

There is not a publicly available dataset of songs, so I will need to acquire them. To do so, I have contacted Spotify but without response as of now. To start, I will try to implement an audio capturing frontend to acquire the songs, without any storage or redistribution of them. If such methods fail, songs from the public domain can be used, or can be purchased through another medium. After compiling a set of songs, the tracks may be preprocessed or split into fixed size chunks in order to yield a better performance from the classifier.

### Neural Network

The largest work of this project is to architect a neural network structure based on previous works that will effectively learn the common characteristic of the dataset that the project intends to transform. To do so there will be multiple trial-and-error iterations to experiment different structures of neural network.

### Presentation

The entire system will most likely include different preprocessors, and a web service for presentation, so that users can transform their piece of music.

### Evaluation

1. Technicality

	Without any human input it is still possible to test the consistency of a neural network. By splitting the dataset into the training set, testing set, and a cross validation set, this project will aim to find a quantitative validation of consistency of the neural network.

2.  Experiment

	I will conduct an experiment on human perception to test that the network has transfered the style of a network as aimed. In order to do so in a controlled manner, experiment will be devised to meet the following standards:

	* To confirm the similarity of content on the dataset, volunteers will be given set of samples of five or more sound tracks, and will be asked to select two that are of the same content. 
	* To confirm the similarity of content on the synthesized data, users will be given the original recording with multiple synthesized sound samples and will be prompted to select the one that has a similar content.
	* To attest to the effectiveness of the transformation, volunteers will be asked to pick one that adheres to the characteristics of the transformation.
	* For consistency throughout the track, different sections of each song may be presented to the volunteers for evaluation.
	* Volunteers will be asked the same questions that are phrased differently, in order to attest to the consistency and reliability.
	* The environment in which the volunteers will be conducting the experiment will also be controlled; all volunteers will be using the same headphones in the same listening environment in order to conduct the experiment.

## Success Criteria

* A successful compilation of music dataset;
* An effective rendering and creation of a neural network architecture;
* Statistically effective behavior of the classifier;
* Successful presentation of the functionalities of the classifier, as described above;
* An agreement of the effectiveness of the classifier by the human perception experiment.
 
## Timetable and Milestones

Since both of my paper 10 modules are in Michaelmas, a larger portion of the project was allocated to Christmas break and Lent Term.

### Until 2 November (4 weeks)

**Preliminary Reading**

* Reading up on deep neural networks and generative adversarial networks, using the textbook by Ian Goodfellow[^8]. 
* Reviewing the papers mentioned above.

**Milestone**

* Write up on the Backgrounds and Introduction section.

### Until 16 November (2 weeks)

* Trying different implementation of the Image Style Transfer, familiarise with the machine learning systems that will be installed on the computing machines.
* Starting to prepare dataset, by contacting music providers, implementing an audio capturing software, or purchasing.

**Milestone**

* Working Image Style Transfer implementation with an in-depth understanding of the network architecture.
* A sizable dataset to start initial experiments.

### Until 30 November (2 weeks)

* Finishing preparation of dataset.
* Initial Experiments on deciding which method may seem more appropriate.

**Milestone**

* A full set of dataset for training, testing, and cross validation.
* A section for the dissertation on methodology chosen and initial experimentation.

### Until 14 December (2 weeks)

* Making a decision on the general architecture of the neural network.
* Validating the choices made and making necessary changes

**Milestone**

* A working initial classifier for further enhancement.

### Until 18 January 2018 (Christmas break, 2 weeks)

* Improving the classifier to start evaluation.
* Preparing the evaluation of the dataset and gathering volunteers.

**Milestone**

* A working transformation engine.
* Fully planned experiment for datasets to be executed in the coming weeks.

### Until 1 February (2 weeks)

* Starting the coding of the presentation code around the transforming network.
* Conducting experiments on the dataset.

**Milestone**

* Results on the experiments on dataset.
* Working presentation layer.

### Until 15 February (2 weeks)

* Using the completed network to prepare experiments for the performance of the software.

**Milestone**

* Full experiment prepared for execution in the coming weeks.

### Until 1 March (2 weeks)

* Conduct experiments prepared and use data in order to start writing up.

**Milestone**

* Initial write up on the implementation and evaluation.

### Until 15 March (2 weeks)

* Writing up the dissertation, completing each part that may be incomplete.

**Milestone**

* Working draft of the write up.

### Until 29 March (Easter break, 2 weeks)

* Finish writing up of the dissertation, and send to supervisors for review.

**Milestone**

* Finished initial draft of the dissertation.

### Until 12 April (Easter break, 2 weeks)
* Start revision on the dissertation after feedback.

**Milestone**

* Revised copy of the dissertation.

### Until 26 April (Easter break, 2 weeks)
* Preparing for submission.

**Milestone**

* Final copy of the dissertation.


