# Neural Graph Collaborative Filtering<sup>∗</sup>

Xiang Wang National University of Singapore xiangwang@u.nus.edu

Xiangnan He† University of Science and Technology of China xiangnanhe@gmail.com

Meng Wang Hefei University of Technology eric.mengwang@gmail.com

Fuli Feng National University of Singapore fulifeng93@gmail.com

Tat-Seng Chua National University of Singapore

# ABSTRACT

Learning vector representations (aka. embeddings) of users and items lies at the core of modern recommender systems. Ranging from early matrix factorization to recently emerged deep learning based methods, existing efforts typically obtain a user's (or an item's) embedding by mapping from pre-existing features that describe the user (or the item), such as ID and attributes. We argue that an inherent drawback of such methods is that, the collaborative signal, which is latent in user-item interactions, is not encoded in the embedding process. As such, the resultant embeddings may not be sufficient to capture the collaborative filtering effect.

In this work, we propose to integrate the user-item interactions more specifically the bipartite graph structure — into the embedding process. We develop a new recommendation framework Neural Graph Collaborative Filtering (NGCF), which exploits the useritem graph structure by propagating embeddings on it. This leads to the expressive modeling of high-order connectivity in useritem graph, effectively injecting the collaborative signal into the embedding process in an explicit manner. We conduct extensive experiments on three public benchmarks, demonstrating significant improvements over several state-of-the-art models like HOP-Rec [\[40\]](#page-9-0) and Collaborative Memory Network [\[5\]](#page-9-1). Further analysis verifies the importance of embedding propagation for learning better user and item representations, justifying the rationality and effectiveness of NGCF. Codes are available at [https://github.com/](https://github.com/xiangwang1223/neural_graph_collaborative_filtering) [xiangwang1223/neural\\_graph\\_collaborative\\_filtering.](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)

## CCS CONCEPTS

#### • Information systems → Recommender systems.

SIGIR '19, July 21–25, 2019, Paris, France

© 2019 Association for Computing Machinery.

ACM ISBN 978-1-4503-6172-9/19/07. . . \$15.00 <https://doi.org/10.1145/3331184.3331267>

dcscts@nus.edu.sg

#### KEYWORDS

Collaborative Filtering, Recommendation, High-order Connectivity, Embedding Propagation, Graph Neural Network

#### ACM Reference Format:

Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019. Neural Graph Collaborative Filtering. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '19), July 21–25, 2019, Paris, France. ACM, New York, NY, USA, [10](#page-9-2) pages.<https://doi.org/10.1145/3331184.3331267>

#### 1 INTRODUCTION

Personalized recommendation is ubiquitous, having been applied to many online services such as E-commerce, advertising, and social media. At its core is estimating how likely a user will adopt an item based on the historical interactions like purchases and clicks. Collaborative filtering (CF) addresses it by assuming that behaviorally similar users would exhibit similar preference on items. To implement the assumption, a common paradigm is to parameterize users and items for reconstructing historical interactions, and predict user preference based on the parameters [\[1,](#page-9-3) [14\]](#page-9-4).

Generally speaking, there are two key components in learnable CF models — 1) embedding, which transforms users and items to vectorized representations, and 2) interaction modeling, which reconstructs historical interactions based on the embeddings. For example, matrix factorization (MF) directly embeds user/item ID as an vector and models user-item interaction with inner product [\[20\]](#page-9-5); collaborative deep learning extends the MF embedding function by integrating the deep representations learned from rich side information of items [\[30\]](#page-9-6); neural collaborative filtering models replace the MF interaction function of inner product with nonlinear neural networks [\[14\]](#page-9-4); and translation-based CF models instead use Euclidean distance metric as the interaction function [\[28\]](#page-9-7), among others.

Despite their effectiveness, we argue that these methods are not sufficient to yield satisfactory embeddings for CF. The key reason is that the embedding function lacks an explicit encoding of the crucial collaborative signal, which is latent in user-item interactions to reveal the behavioral similarity between users (or items). To be more specific, most existing methods build the embedding function with the descriptive features only (e.g., ID and attributes), without considering the user-item interactions — which are only used to define the objective function for model training [\[26,](#page-9-8) [28\]](#page-9-7). As a result,

<sup>∗</sup> In the version published in ACM Digital Library, we find some small bugs; the bugs do not change the comparison results and the empirical findings. In this latest version, we update and correct the experimental results (i.e., the preprocessing of Yelp2018 dataset and the ndcg metric). All updates are highlighted in footnotes. †Xiangnan He is the corresponding author.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than ACM must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

<span id="page-1-0"></span>![](_page_1_Figure_0.jpeg)

Figure 1: An illustration of the user-item interaction graph and the high-order connectivity. The node <sup>u</sup><sup>1</sup> is the target user to provide recommendations for.

when the embeddings are insufficient in capturing CF, the methods have to rely on the interaction function to make up for the deficiency of suboptimal embeddings [\[14\]](#page-9-4).

While intuitively useful to integrate user-item interactions into the embedding function, it is non-trivial to do it well. In particular, the scale of interactions can easily reach millions or even larger in real applications, making it difficult to distill the desired collaborative signal. In this work, we tackle the challenge by exploiting the high-order connectivity from useritem interactions, a natural way that encodes collaborative signal in the interaction graph structure.

Running Example. Figure [1](#page-1-0) illustrates the concept of high-order connectivity. The user of interest for recommendation is u1, labeled with the double circle in the left subfigure of user-item interaction graph. The right subfigure shows the tree structure that is expanded from u1. The high-order connectivity denotes the path that reaches <sup>u</sup><sup>1</sup> from any node with the path length <sup>l</sup> larger than 1. Such highorder connectivity contains rich semantics that carry collaborative signal. For example, the path <sup>u</sup><sup>1</sup> <sup>←</sup> <sup>i</sup><sup>2</sup> <sup>←</sup> <sup>u</sup><sup>2</sup> indicates the behavior similarity between <sup>u</sup><sup>1</sup> and <sup>u</sup>2, as both users have interacted with <sup>i</sup>2; the longer path <sup>u</sup><sup>1</sup> <sup>←</sup> <sup>i</sup><sup>2</sup> <sup>←</sup> <sup>u</sup><sup>2</sup> <sup>←</sup> <sup>i</sup><sup>4</sup> suggests that <sup>u</sup><sup>1</sup> is likely to adopti4, since her similar useru<sup>2</sup> has consumed <sup>i</sup><sup>4</sup> before. Moreover, from the holistic view of <sup>l</sup> = 3, item <sup>i</sup><sup>4</sup> is more likely to be of interest to <sup>u</sup><sup>1</sup> than item <sup>i</sup>5, since there are two paths connecting <i4,u1>, while only one path connects <i5,u1>.

Present Work. We propose to model the high-order connectivity information in the embedding function. Instead of expanding the interaction graph as a tree which is complex to implement, we design a neural network method to propagate embeddings recursively on the graph. This is inspired by the recent developments of graph neural networks [\[8,](#page-9-9) [32,](#page-9-10) [38\]](#page-9-11), which can be seen as constructing information flows in the embedding space. Specifically, we devise an embedding propagation layer, which refines a user's (or an item's) embedding by aggregating the embeddings of the interacted items (or users). By stacking multiple embedding propagation layers, we can enforce the embeddings to capture the collaborative signal in high-order connectivities. Taking Figure [1](#page-1-0) as an example, stacking two layers captures the behavior similarity of <sup>u</sup><sup>1</sup> <sup>←</sup> <sup>i</sup><sup>2</sup> <sup>←</sup> <sup>u</sup>2, stacking three layers captures the potential recommendations of <sup>u</sup><sup>1</sup> <sup>←</sup> <sup>i</sup><sup>2</sup> <sup>←</sup> <sup>u</sup><sup>2</sup> <sup>←</sup> <sup>i</sup>4, and the strength of the information flow (which is estimated by the trainable weights between layers) determines the recommendation priority of <sup>i</sup><sup>4</sup> and <sup>i</sup>5. We conduct extensive experiments on three public

benchmarks to verify the rationality and effectiveness of our Neural Graph Collaborative Filtering (NGCF) method.

Lastly, it is worth mentioning that although the high-order connectivity information has been considered in a very recent method named HOP-Rec [\[40\]](#page-9-0), it is only exploited to enrich the training data. Specifically, the prediction model of HOP-Rec remains to be MF, while it is trained by optimizing a loss that is augmented with high-order connectivities. Distinct from HOP-Rec, we contribute a new technique to integrate high-order connectivities into the prediction model, which empirically yields better embeddings than HOP-Rec for CF.

To summarize, this work makes the following main contributions:

- We highlight the critical importance of explicitly exploiting the collaborative signal in the embedding function of model-based CF methods.
- We propose NGCF, a new recommendation framework based on graph neural network, which explicitly encodes the collaborative signal in the form of high-order connectivities by performing embedding propagation.
- We conduct empirical studies on three million-size datasets. Extensive results demonstrate the state-of-the-art performance of NGCF and its effectiveness in improving the embedding quality with neural embedding propagation.

#### 2 METHODOLOGY

We now present the proposed NGCF model, the architecture of which is illustrated in Figure [2.](#page-2-0) There are three components in the framework: (1) an embedding layer that offers and initialization of user embeddings and item embeddings; (2) multiple embedding propagation layers that refine the embeddings by injecting highorder connectivity relations; and (3) the prediction layer that aggregates the refined embeddings from different propagation layers and outputs the affinity score of a user-item pair. Finally, we discuss the time complexity of NGCF and the connections with existing methods.

#### 2.1 Embedding Layer

Following mainstream recommender models [\[1,](#page-9-3) [14,](#page-9-4) [26\]](#page-9-8), we describe a user <sup>u</sup> (an item <sup>i</sup>) with an embedding vector <sup>e</sup><sup>u</sup> <sup>∈</sup> <sup>R</sup> d (e<sup>i</sup> ∈ R d ), where d denotes the embedding size. This can be seen as building a parameter matrix as an embedding look-up table:

$$
E = [\underbrace{e_{u_1}, \cdots, e_{u_N}}_{\text{sum with all time, it is not all in.}}].
$$
 (1)

users embeddings item embeddings

It is worth noting that this embedding table serves as an initial state for user embeddings and item embeddings, to be optimized in an end-to-end fashion. In traditional recommender models like MF and neural collaborative filtering [\[14\]](#page-9-4), these ID embeddings are directly fed into an interaction layer (or operator) to achieve the prediction score. In contrast, in our NGCF framework, we refine the embeddings by propagating them on the user-item interaction graph. This leads to more effective embeddings for recommendation, since the embedding refinement step explicitly injects collaborative signal into embeddings.

<span id="page-2-0"></span>![](_page_2_Figure_0.jpeg)

Figure 2: An illustration of NGCF model architecture (the arrowed lines present the flow of information). The representations of useru<sup>1</sup> (left) and itemi<sup>4</sup> (right) are refined with multiple embedding propagation layers, whose outputs are concatenated to make the final prediction.

#### 2.2 Embedding Propagation Layers

Next we build upon the message-passing architecture of GNNs [\[8,](#page-9-9) [38\]](#page-9-11) in order to capture CF signal along the graph structure and refine the embeddings of users and items. We first illustrate the design of one-layer propagation, and then generalize it to multiple successive layers.

2.2.1 First-order Propagation. Intuitively, the interacted items provide direct evidence on a user's preference [\[16,](#page-9-12) [39\]](#page-9-13); analogously, the users that consume an item can be treated as the item's features and used to measure the collaborative similarity of two items. We build upon this basis to perform embedding propagation between the connected users and items, formulating the process with two major operations: message construction and message aggregation.

Message Construction. For a connected user-item pair (u,i), we define the message from i to u as:

$$
\mathbf{m}_{u \leftarrow i} = f(\mathbf{e}_i, \mathbf{e}_u, p_{ui}), \tag{2}
$$

where mu←i is the message embedding (i.e., the information to be propagated). f (·) is the message encoding function, which takes embeddings <sup>e</sup><sup>i</sup> and <sup>e</sup><sup>u</sup> as input, and uses the coefficient <sup>p</sup>ui to control the decay factor on each propagation on edge (u,i).

In this work, we implement f (·) as:

$$
\mathbf{m}_{u \leftarrow i} = \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} \Big(\mathbf{W}_1 \mathbf{e}_i + \mathbf{W}_2 (\mathbf{e}_i \odot \mathbf{e}_u) \Big), \tag{3}
$$

where <sup>W</sup>1, <sup>W</sup><sup>2</sup> <sup>∈</sup> <sup>R</sup> d ′×d are the trainable weight matrices to distill useful information for propagation, and d ′ is the transformation size. Distinct from conventional graph convolution networks [\[4,](#page-9-14) [18,](#page-9-15) [29,](#page-9-16) [42\]](#page-9-17) that consider the contribution of ei only, here we additionally encode the interaction between ei and eu into the message being passed via e<sup>i</sup> ⊙ e<sup>u</sup> , where ⊙ denotes the element-wise product. This makes the message dependent on the affinity between ei and eu , e.g., passing more messages from the similar items. This not

<span id="page-2-2"></span>![](_page_2_Figure_11.jpeg)

Figure 3: Illustration of third-order embedding propagation for user u1. Best view in color.

only increases the model representation ability, but also boosts the performance for recommendation (evidences in our experiments Section [4.4.2\)](#page-7-0).

Following the graph convolutional network [\[18\]](#page-9-15), we set <sup>p</sup>ui as the graph Laplacian norm <sup>1</sup>/ p |N<sup>u</sup> ||N<sup>i</sup> |, where N<sup>u</sup> and N<sup>i</sup> denote the first-hop neighbors of user u and item i. From the viewpoint of representation learning, <sup>p</sup>ui reflects how much the historical item contributes the user preference. From the viewpoint of message passing, <sup>p</sup>ui can be interpreted as a discount factor, considering the messages being propagated should decay with the path length.

Message Aggregation. In this stage, we aggregate the messages propagated from u's neighborhood to refine u's representation. Specifically, we define the aggregation function as:

$$
\mathbf{e}_u^{(1)} = \text{LeakyReLU}\Big(\mathbf{m}_{u \leftarrow u} + \sum_{i \in \mathcal{N}_u} \mathbf{m}_{u \leftarrow i}\Big),\tag{4}
$$

where e (1) <sup>u</sup> denotes the representation of user <sup>u</sup> obtained after the first embedding propagation layer. The activation function of LeakyReLU [\[23\]](#page-9-18) allows messages to encode both positive and small negative signals. Note that in addition to the messages propagated from neighbors <sup>N</sup><sup>u</sup> , we take the self-connection of <sup>u</sup> into consideration: mu←u = W1eu , which retains the information of original features (W1 is the weight matrix shared with the one used in Equation [\(3\)](#page-2-1)). Analogously, we can obtain the representation e (1) i for item i by propagating information from its connected users. To summarize, the advantage of the embedding propagation layer lies in explicitly exploiting the first-order connectivity information to relate user and item representations.

<span id="page-2-1"></span>2.2.2 High-order Propagation. With the representations augmented by first-order connectivity modeling, we can stack more embedding propagation layers to explore the high-order connectivity information. Such high-order connectivities are crucial to encode the collaborative signal to estimate the relevance score between a user and item.

By stacking l embedding propagation layers, a user (and an item) is capable of receiving the messages propagated from its l-hop neighbors. As Figure [2](#page-2-0) displays, in the l-th step, the representation of user u is recursively formulated as:

<span id="page-2-3"></span>
$$
\mathbf{e}_{u}^{(l)} = \text{LeakyReLU}\Big(\mathbf{m}_{u \leftarrow u}^{(l)} + \sum_{i \in \mathcal{N}_{u}} \mathbf{m}_{u \leftarrow i}^{(l)}\Big),\tag{5}
$$

wherein the messages being propagated are defined as follows,

$$
\begin{cases}\n\mathbf{m}_{u \leftarrow i}^{(l)} = p_{ui} \left( \mathbf{W}_1^{(l)} \mathbf{e}_i^{(l-1)} + \mathbf{W}_2^{(l)} (\mathbf{e}_i^{(l-1)} \odot \mathbf{e}_u^{(l-1)}) \right), \\
\mathbf{m}_{u \leftarrow u}^{(l)} = \mathbf{W}_1^{(l)} \mathbf{e}_u^{(l-1)},\n\end{cases} \tag{6}
$$

where W (l) 1 , W (l) 2 , <sup>∈</sup> <sup>R</sup> <sup>d</sup><sup>l</sup> <sup>×</sup>dl−<sup>1</sup> are the trainable transformation matrices, and <sup>d</sup><sup>l</sup> is the transformation size; e (l−1) i is the item representation generated from the previous message-passing steps, memorizing the messages from its (l-1)-hop neighbors. It further contributes to the representation of user u at layer l. Analogously, we can obtain the representation for item i at the layer l.

As Figure [3](#page-2-2) shows, the collaborative signal like <sup>u</sup><sup>1</sup> <sup>←</sup> <sup>i</sup><sup>2</sup> <sup>←</sup> <sup>u</sup><sup>2</sup> <sup>←</sup> <sup>i</sup><sup>4</sup> can be captured in the embedding propagation process. Furthermore, the message from <sup>i</sup><sup>4</sup> is explicitly encoded in <sup>e</sup> (3) u1 (indicated by the red line). As such, stacking multiple embedding propagation layers seamlessly injects collaborative signal into the representation learning process.

Propagation Rule in Matrix Form. To offer a holistic view of embedding propagation and facilitate batch implementation, we provide the matrix form of the layer-wise propagation rule (equivalent to Equations [\(5\)](#page-2-3) and [\(6\)](#page-3-0)):

$$
\mathbf{E}^{(l)} = \text{LeakyReLU}\Big((\mathcal{L} + \mathbf{I})\mathbf{E}^{(l-1)}\mathbf{W}_1^{(l)} + \mathcal{L}\mathbf{E}^{(l-1)} \odot \mathbf{E}^{(l-1)}\mathbf{W}_2^{(l)}\Big), (7)
$$

where E (l) ∈ R (<sup>N</sup> <sup>+</sup>M)×d<sup>l</sup> are the representations for users and items obtained after l steps of embedding propagation. <sup>E</sup> (0) is set as E at the initial message-passing iteration, that is e (0) <sup>u</sup> = e<sup>u</sup> and e (0) i = ei ; and I denote an identity matrix. L represents the Laplacian matrix for the user-item graph, which is formulated as:

$$
\mathcal{L} = \mathbf{D}^{-\frac{1}{2}} \mathbf{A} \mathbf{D}^{-\frac{1}{2}} \text{ and } \mathbf{A} = \begin{bmatrix} \mathbf{0} & \mathbf{R} \\ \mathbf{R}^{\top} & \mathbf{0} \end{bmatrix},\tag{8}
$$

where <sup>R</sup> <sup>∈</sup> R <sup>N</sup> <sup>×</sup><sup>M</sup> is the user-item interaction matrix, and 0 is allzero matrix; A is the adjacency matrix and D is the diagonal degree matrix, where the <sup>t</sup>-th diagonal element <sup>D</sup>t t <sup>=</sup> |N<sup>t</sup> |; as such, the nonzero off-diagonal entry <sup>L</sup>ui = 1/ p |N<sup>u</sup> ||N<sup>i</sup> |, which is equal to <sup>p</sup>ui used in Equation [\(3\)](#page-2-1).

By implementing the matrix-form propagation rule, we can simultaneously update the representations for all users and items in a rather efficient way. It allows us to discard the node sampling procedure, which is commonly used to make graph convolution network runnable on large-scale graph [\[25\]](#page-9-19). We will analyze the complexity in Section [2.5.2.](#page-4-0)

#### 2.3 Model Prediction

After propagating with L layers, we obtain multiple representations for user u, namely {<sup>e</sup> (1) <sup>u</sup> , · · · , <sup>e</sup> (L) <sup>u</sup> }. Since the representations obtained in different layers emphasize the messages passed over different connections, they have different contributions in reflecting user preference. As such, we concatenate them to constitute the final embedding for a user; we do the same operation on items, concatenating the item representations {e (1) i , · · · , <sup>e</sup> (L) i } learned by different layers to get the final item embedding:

$$
\mathbf{e}_u^* = \mathbf{e}_u^{(0)} || \cdots || \mathbf{e}_u^{(L)}, \quad \mathbf{e}_i^* = \mathbf{e}_i^{(0)} || \cdots || \mathbf{e}_i^{(L)}, \tag{9}
$$

<span id="page-3-0"></span>where ∥ is the concatenation operation. By doing so, we not only enrich the initial embeddings with embedding propagation layers, but also allow controlling the range of propagation by adjusting L. Note that besides concatenation, other aggregators can also be applied, such as weighted average, max pooling, LSTM, etc., which imply different assumptions in combining the connectivities of different orders. The advantage of using concatenation lies in its simplicity, since it involves no additional parameters to learn, and it has been shown quite effectively in a recent work of graph neural networks [\[38\]](#page-9-11), which refers to layer-aggregation mechanism.

Finally, we conduct the inner product to estimate the user's preference towards the target item:

$$
\hat{y}_{\text{NGCF}}(u,i) = \mathbf{e}_u^* \mathbf{e}_i^* \,. \tag{10}
$$

In this work, we emphasize the embedding function learning thus only employ the simple interaction function of inner product. Other more complicated choices, such as neural network-based interaction functions [\[14\]](#page-9-4), are left to explore in the future work.

#### 2.4 Optimization

To learn model parameters, we optimize the pairwise BPR loss [\[26\]](#page-9-8), which has been intensively used in recommender systems [\[2,](#page-9-20) [13\]](#page-9-21). It considers the relative order between observed and unobserved user-item interactions. Specifically, BPR assumes that the observed interactions, which are more reflective of a user's preferences, should be assigned higher prediction values than unobserved ones. The objective function is as follows,

<span id="page-3-1"></span>
$$
Loss = \sum_{(u,i,j)\in O} -\ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda ||\Theta||_2^2, \qquad (11)
$$

where <sup>O</sup> <sup>=</sup> {(u,i, j)|(u,i) ∈ R<sup>+</sup> , (u, j) ∈ R−} denotes the pairwise training data, R + indicates the observed interactions, and R <sup>−</sup> is the unobserved interactions; σ(·) is the sigmoid function; Θ = {E, {<sup>W</sup> (l) 1 , W (l) 2 } L <sup>l</sup>=1} denotes all trainable model parameters, and λ controls the <sup>L</sup><sup>2</sup> regularization strength to prevent overfitting. We adopt mini-batch Adam [\[17\]](#page-9-22) to optimize the prediction model and update the model parameters. In particular, for a batch of randomly sampled triples (u,i, j) ∈ O, we establish their representations [e (0) , · · · , <sup>e</sup> (L) ] after L steps of propagation, and then update model parameters by using the gradients of the loss function.

2.4.1 Model Size. It is worth pointing out that although NGCF obtains an embedding matrix (E (l) ) at each propagation layer l, it only introduces very few parameters — two weight matrices of size <sup>d</sup><sup>l</sup> <sup>×</sup> <sup>d</sup>l−<sup>1</sup> . Specifically, these embedding matrices are derived from the embedding look-up table E (0), with the transformation based on the user-item graph structure and weight matrices. As such, compared to MF — the most concise embeddingbased recommender model, our NGCF uses only <sup>2</sup>Ldldl−<sup>1</sup> more parameters. Such additional cost on model parameters is almost negligible, considering that L is usually a number smaller than 5, and <sup>d</sup><sup>l</sup> is typically set as the embedding size, which is much smaller than the number of users and items. For example, on our experimented Gowalla dataset (20K users and 40K items), when the embedding size is 64 and we use 3 propagation layers of size 64×64, MF has 4.5 million parameters, while our NGCF uses only 0.024 million additional parameters. To summarize, NGCF uses very few

additional model parameters to achieve the high-order connectivity modeling.

2.4.2 Message and Node Dropout. Although deep learning models have strong representation ability, they usually suffer from overfitting. Dropout is an effective solution to prevent neural networks from overfitting. Following the prior work on graph convolutional network [\[29\]](#page-9-16), we propose to adopt two dropout techniques in NGCF: message dropout and node dropout. Message dropout randomly drops out the outgoing messages. Specifically, we drop out the messages being propagated in Equation [\(6\)](#page-3-0), with a probability p1. As such, in the l-th propagation layer, only partial messages contribute to the refined representations. We also conduct node dropout to randomly block a particular node and discard all its outgoing messages. For the l-th propagation layer, we randomly drop (<sup>M</sup> <sup>+</sup> <sup>N</sup>)p<sup>2</sup> nodes of the Laplacian matrix, where <sup>p</sup><sup>2</sup> is the dropout ratio.

Note that dropout is only used in training, and must be disabled during testing. The message dropout endows the representations more robustness against the presence or absence of single connections between users and items, and the node dropout focuses on reducing the influences of particular users or items. We perform experiments to investigate the impact of message dropout and node dropout on NGCF in Section [4.4.3.](#page-7-1)

#### 2.5 Discussions

In the subsection, we first show how NGCF generalizes SVD++ [\[19\]](#page-9-23). In what follows, we analyze the time complexity of NGCF.

2.5.1 NGCF Generalizes SVD++. SVD++ can be viewed as a special case of NGCF with no high-order propagation layer. In particular, we set L to one. Within the propagation layer, we disable the transformation matrix and nonlinear activation function. Thereafter, e (1) <sup>u</sup> and e (1) i are treated as the final representations for user u and item i, respectively. We term this simplified model as NGCF-SVD, which can be formulated as:

$$
\hat{y}_{\text{NGCF-SVD}} = \left(\mathbf{e}_u + \sum_{i' \in \mathcal{N}_u} p_{ui'} \mathbf{e}_{i'}\right)^{\top} \left(\mathbf{e}_i + \sum_{u' \in \mathcal{N}_i} p_{iu'} \mathbf{e}_i\right). \tag{12}
$$

Clearly, by setting <sup>p</sup>ui′ and <sup>p</sup><sup>u</sup> ′ <sup>i</sup> as <sup>1</sup>/ p |N<sup>u</sup> | and 0 separately, we can exactly recover SVD++ model. Moreover, another widely-used item-based CF model, FISM [\[16\]](#page-9-12), can be also seen as a special case of NGCF, wherein piu′ in Equation [\(12\)](#page-4-1) is set as 0.

<span id="page-4-0"></span>2.5.2 Time Complexity Analysis. As we can see, the layer-wise propagation rule is the main operation. For the l-th propagation layer, the matrix multiplication has computational complexity O(|R<sup>+</sup> <sup>|</sup>dldl−<sup>1</sup> ), where |R<sup>+</sup> | denotes the number of nonzero entires in the Laplacian matrix; and <sup>d</sup><sup>l</sup> and <sup>d</sup>l−<sup>1</sup> are the current and previous transformation size. For the prediction layer, only the inner product is involved, for which the time complexity of the whole training epoch is O( P<sup>L</sup> <sup>l</sup>=1 |R<sup>+</sup> |dl ). Therefore, the overall complexity for evaluating NGCF is O( P<sup>L</sup> <sup>l</sup>=1 |R<sup>+</sup> <sup>|</sup>dldl−<sup>1</sup> <sup>+</sup> P<sup>L</sup> <sup>l</sup>=1 |R<sup>+</sup> |dl ). Empirically, under the same experimental settings (as explained in Section [4\)](#page-5-0), MF and NGCF cost around <sup>20</sup>s and <sup>80</sup>s per epoch on Gowalla dataset for training, respectively; during inference, the time costs of MF and NGCF are <sup>80</sup>s and <sup>260</sup>s for all testing instances, respectively.

#### 3 RELATED WORK

We review existing work on model-based CF, graph-based CF, and graph neural network-based methods, which are most relevant with this work. Here we highlight the differences with our NGCF.

#### 3.1 Model-based CF Methods

Modern recommender systems [\[5,](#page-9-1) [14,](#page-9-4) [33\]](#page-9-24) parameterize users and items by vectorized representations and reconstruct user-item interaction data based on model parameters. For example, MF [\[20,](#page-9-5) [26\]](#page-9-8) projects the ID of each user and item as an embedding vector, and conducts inner product between them to predict an interaction. To enhance the embedding function, much effort has been devoted to incorporate side information like item content [\[2,](#page-9-20) [30\]](#page-9-6), social relations [\[34\]](#page-9-25), item relations [\[37\]](#page-9-26), user reviews [\[3\]](#page-9-27), and external knowledge graph [\[32,](#page-9-10) [35\]](#page-9-28). While inner product can force user and item embeddings of an observed interaction close to each other, its linearity makes it insufficient to reveal the complex and nonlinear relationships between users and items [\[14,](#page-9-4) [15\]](#page-9-29). Towards this end, recent efforts [\[11,](#page-9-30) [14,](#page-9-4) [15,](#page-9-29) [36\]](#page-9-31) focus on exploiting deep learning techniques to enhance the interaction function, so as to capture the nonlinear feature interactions between users and items. For instance, neural CF models, such as NeuMF [\[14\]](#page-9-4), employ nonlinear neural networks as the interaction function; meanwhile, translationbased CF models, such as LRML [\[28\]](#page-9-7), instead model the interaction strength with Euclidean distance metrics.

Despite great success, we argue that the design of the embedding function is insufficient to yield optimal embeddings for CF, since the CF signals are only implicitly captured. Summarizing these methods, the embedding function transforms the descriptive features (e.g., ID and attributes) to vectors, while the interaction function serves as a similarity measure on the vectors. Ideally, when user-item interactions are perfectly reconstructed, the transitivity property of behavior similarity could be captured. However, such transitivity effect showed in the Running Example is not explicitly encoded, thus there is no guarantee that the indirectly connected users and items are close in the embedding space. Without an explicit encoding of the CF signals, it is hard to obtain embeddings that meet the desired properties.

#### <span id="page-4-1"></span>3.2 Graph-Based CF Methods

Another line of research [\[12,](#page-9-32) [24,](#page-9-33) [40\]](#page-9-0) exploits the user-item interaction graph to infer user preference. Early efforts, such as ItemRank [\[7\]](#page-9-34) and BiRank [\[12\]](#page-9-32), adopt the idea of label propagation to capture the CF effect. To score items for a user, these methods define the labels as her interacted items, and propagate the labels on the graph. As the recommendation scores are obtained based on the structural reachness (which can be seen as a kind of similarity) between the historical items and the target item, these methods essentially belong to neighbor-based methods. However, these methods are conceptually inferior to model-based CF methods, since there lacks model parameters to optimize the objective function of recommendation.

The recently proposed method HOP-Rec [\[40\]](#page-9-0) alleviates the problem by combining graph-based with embedding-based method. It first performs random walks to enrich the interactions of a user with multi-hop connected items. Then it trains MF with BPR

| Table 1: Statistics of the datasets. |  |
|--------------------------------------|--|
|                                      |  |

<span id="page-5-3"></span>

| Dataset     | #Users  | #Items  | #Interactions | Density |  |
|-------------|---------|---------|---------------|---------|--|
| Gowalla     | 29, 858 | 40, 981 | 1, 027, 370   | 0.00084 |  |
| Yelp2018∗   | 31, 668 | 38, 048 | 1, 561, 406   | 0.00130 |  |
| Amazon-Book | 52, 643 | 91, 599 | 2, 984, 108   | 0.00062 |  |
|             |         |         |               |         |  |

objective based on the enriched user-item interaction data to build the recommender model. The superior performance of HOP-Rec over MF provides evidence that incorporating the connectivity information is beneficial to obtain better embeddings in capturing the CF effect. However, we argue that HOP-Rec does not fully explore the high-order connectivity, which is only utilized to enrich the training data[1](#page-5-1) , rather than directly contributing to the model's embedding function. Moreover, the performance of HOP-Rec depends heavily on the random walks, which require careful tuning efforts such as a proper setting of decay factor.

#### 3.3 Graph Convolutional Networks

By devising a specialized graph convolution operation on useritem interaction graph (cf. Equation [\(3\)](#page-2-1)), we make NGCF effective in exploiting the CF signal in high-order connectivities. Here we discuss existing recommendation methods that also employ graph convolution operations [\[29,](#page-9-16) [42,](#page-9-17) [43\]](#page-9-35).

GC-MC [\[29\]](#page-9-16) applies the graph convolution network (GCN) [\[18\]](#page-9-15) on user-item graph, however it only employs one convolutional layer to exploit the direct connections between users and items. Hence it fails to reveal collaborative signal in high-order connectivities. PinSage [\[42\]](#page-9-17) is an industrial solution that employs multiple graph convolution layers on item-item graph for Pinterest image recommendation. As such, the CF effect is captured on the level of item relations, rather than the collective user behaviors. SpectralCF [\[43\]](#page-9-35) proposes a spectral convolution operation to discover all possible connectivity between users and items in the spectral domain. Through the eigen-decomposition of graph adjacency matrix, it can discover the connections between a user-item pair. However, the eigen-decomposition causes a high computational complexity, which is very time-consuming and difficult to support large-scale recommendation scenarios.

#### <span id="page-5-0"></span>4 EXPERIMENTS

We perform experiments on three real-world datasets to evaluate our proposed method, especially the embedding propagation layer. We aim to answer the following research questions:

- RQ1: How does NGCF perform as compared with state-of-the-art CF methods?
- RQ2: How do different hyper-parameter settings (e.g., depth of layer, embedding propagation layer, layer-aggregation mechanism, message dropout, and node dropout) affect NGCF?
- RQ3: How do the representations benefit from the high-order connectivity?

#### 4.1 Dataset Description

To evaluate the effectiveness of NGCF, we conduct experiments on three benchmark datasets: Gowalla, Yelp2018∗[2](#page-5-2) , and Amazon-book, which are publicly accessible and vary in terms of domain, size, and sparsity. We summarize the statistics of three datasets in Table [1.](#page-5-3)

Gowalla: This is the check-in dataset [\[21\]](#page-9-36) obtained from Gowalla, where users share their locations by checking-in. To ensure the quality of the dataset, we use the 10-core setting [\[10\]](#page-9-37), i.e., retaining users and items with at least ten interactions.

Yelp2018<sup>∗</sup> : This dataset is adopted from the 2018 edition of the Yelp challenge. Wherein, the local businesses like restaurants and bars are viewed as the items. We use the same 10-core setting in order to ensure data quality.

Amazon-book: Amazon-review is a widely used dataset for product recommendation [\[9\]](#page-9-38). We select Amazon-book from the collection. Similarly, we use the 10-core setting to ensure that each user and item have at least ten interactions.

For each dataset, we randomly select 80% of historical interactions of each user to constitute the training set, and treat the remaining as the test set. From the training set, we randomly select 10% of interactions as validation set to tune hyper-parameters. For each observed user-item interaction, we treat it as a positive instance, and then conduct the negative sampling strategy to pair it with one negative item that the user did not consume before.

### 4.2 Experimental Settings

4.2.1 Evaluation Metrics. For each user in the test set, we treat all the items that the user has not interacted with as the negative items. Then each method outputs the user's preference scores over all the items, except the positive ones used in the training set. To evaluate the effectiveness of top-K recommendation and preference ranking, we adopt two widely-used evaluation protocols [\[14,](#page-9-4) [40\]](#page-9-0): recall@K and ndcg@K [3](#page-5-4) . By default, we set K = 20. We report the average metrics for all users in the test set.

4.2.2 Baselines. To demonstrate the effectiveness, we compare our proposed NGCF with the following methods:

- MF [\[26\]](#page-9-8): This is matrix factorization optimized by the Bayesian personalized ranking (BPR) loss, which exploits the user-item direct interactions only as the target value of interaction function.
- NeuMF [\[14\]](#page-9-4): The method is a state-of-the-art neural CF model which uses multiple hidden layers above the element-wise and concatenation of user and item embeddings to capture their nonlinear feature interactions. Especially, we employ two-layered plain architecture, where the dimension of each hidden layer keeps the same.
- CMN [\[5\]](#page-9-1): It is a state-of-the-art memory-based model, where the user representation attentively combines the memory slots of neighboring users via the memory layers. Note that the firstorder connections are used to find similar users who interacted with the same items.
- HOP-Rec [\[40\]](#page-9-0): This is a state-of-the-art graph-based model, where the high-order neighbors derived from random walks are exploited to enrich the user-item interaction data.
- PinSage [\[42\]](#page-9-17): PinSage is designed to employ GraphSAGE [\[8\]](#page-9-9) on item-item graph. In this work, we apply it on user-item

<span id="page-5-2"></span><span id="page-5-1"></span><sup>1</sup>The enriched trained data can be seen as a regularizer to the original training. 2 In the previous version of yelp2018, we did not filter out cold-start items in the testing set, and hence we rerun all methods.

<span id="page-5-4"></span><sup>3</sup>The previous implementation of ndcg metric in NGCF follows the codes [\[31\]](#page-9-39), but we find it is slightly different from the standard definition, although reflecting the similar trends. We re-implement the ndcg metric and rerun all methods

<span id="page-6-1"></span>

|          | Gowalla |         |         | Yelp2018∗ | Amazon-Book |         |
|----------|---------|---------|---------|-----------|-------------|---------|
|          | recall  | ndcg    | recall  | ndcg      | recall      | ndcg    |
| MF       | 0.1291  | 0.1109  | 0.0433  | 0.0354    | 0.0250      | 0.0196  |
| NeuMF    | 0.1399  | 0.1212  | 0.0451  | 0.0363    | 0.0258      | 0.0200  |
| CMN      | 0.1405  | 0.1221  | 0.0457  | 0.0369    | 0.0267      | 0.0218  |
| HOP-Rec  | 0.1399  | 0.1214  | 0.0517  | 0.0428    | 0.0309      | 0.0232  |
| GC-MC    | 0.1395  | 0.1204  | 0.0462  | 0.0379    | 0.0288      | 0.0224  |
| PinSage  | 0.1380  | 0.1196  | 0.0471  | 0.0393    | 0.0282      | 0.0219  |
| NGCF-3   | 0.1569∗ | 0.1327∗ | 0.0579∗ | 0.0477∗   | 0.0337∗     | 0.0261∗ |
| %Improv. | 11.68%  | 8.64%   | 11.97%  | 11.29%    | 9.61%       | 12.50%  |
| p-value  | 2.01e-7 | 3.03e-3 | 5.34e-3 | 4.62e-4   | 3.48e-5     | 1.26e-4 |
|          |         |         |         |           |             |         |

Table 2: Overall Performance Comparison.

interaction graph. Especially, we employ two graph convolution layers as suggested in [\[42\]](#page-9-17), and the hidden dimension is set equal to the embedding size.

• GC-MC [\[29\]](#page-9-16): This model adopts GCN [\[18\]](#page-9-15) encoder to generate the representations for users and items, where only the first-order neighbors are considered. Hence one graph convolution layer, where the hidden dimension is set as the embedding size, is used as suggested in [\[29\]](#page-9-16).

We also tried SpectralCF [\[43\]](#page-9-35) but found that the eigen-decomposition leads to high time cost and resource cost, especially when the number of users and items is large. Hence, although it achieved promising performance in small datasets, we did not select it for comparison. For fair comparison, all methods optimize the BPR loss as shown in Equation [\(11\)](#page-3-1).

4.2.3 Parameter Settings. We implement our NGCF model in Tensorflow. The embedding size is fixed to 64 for all models. For HOP-Rec, we search the steps of random walks in {1, <sup>2</sup>, <sup>3</sup>} and tune the learning rate in {0.025, <sup>0</sup>.020, <sup>0</sup>.015, <sup>0</sup>.010}. We optimize all models except HOP-Rec with the Adam optimizer, where the batch size is fixed at 1024. In terms of hyperparameters, we apply a grid search for hyperparameters: the learning rate is tuned amongst {0.0001, <sup>0</sup>.0005, <sup>0</sup>.001, <sup>0</sup>.005}, the coefficient of <sup>L</sup><sup>2</sup> normalization is searched in {10−<sup>5</sup> , <sup>10</sup>−<sup>4</sup> , · · · , <sup>10</sup><sup>1</sup> , 102 }, and the dropout ratio in {0.0, <sup>0</sup>.1, · · · , <sup>0</sup>.8}. Besides, we employ the node dropout technique for GC-MC and NGCF, where the ratio is tuned in {0.0, <sup>0</sup>.1, · · · , <sup>0</sup>.8}. We use the Xavier initializer [\[6\]](#page-9-40) to initialize the model parameters[4](#page-6-0) . Moreover, early stopping strategy is performed, i.e., premature stopping if recall@20 on the validation data does not increase for 50 successive epochs. To model the CF signal encoded in thirdorder connectivity, we set the depth of NGCF L as three. Without specification, we show the results of three embedding propagation layers, node dropout ratio of <sup>0</sup>.0, and message dropout ratio of <sup>0</sup>.1.

#### 4.3 Performance Comparison (RQ1)

We start by comparing the performance of all the methods, and then explore how the modeling of high-order connectivity improves under the sparse settings.

4.3.1 Overall Comparison. Table [2](#page-6-1) reports the performance comparison results. We have the following observations:

• MF achieves poor performance on three datasets. This indicates that the inner product is insufficient to capture the complex relations between users and items, further limiting the performance. NeuMF consistently outperforms MF across all cases, demonstrating the importance of nonlinear feature interactions between user and item embeddings. However, neither MF nor NeuMF explicitly models the connectivity in the embedding learning process, which could easily lead to suboptimal representations.

- Compared to MF and NeuMF, the performance of GC-MC verifies that incorporating the first-order neighbors can improve the representation learning. However, in Yelp2018<sup>∗</sup> , GC-MC underperforms NeuMF w.r.t. ndcg@20. The reason might be that GC-MC fails to fully explore the nonlinear feature interactions between users and items.
- CMN generally achieves better performance than GC-MC in most cases. Such improvement might be attributed to the neural attention mechanism, which can specify the attentive weight of each neighboring user, rather than the equal or heuristic weight used in GC-MC.
- PinSage slightly underperforms CMN in Gowalla and Amazon-Book, while performing much better in Yelp2018<sup>∗</sup> ; meanwhile, HOP-Rec generally achieves remarkable improvements in most cases. It makes sense since PinSage introduces high-order connectivity in the embedding function, and HOP-Rec exploits high-order neighbors to enrich the training data, while CMN considers the similar users only. It therefore points to the positive effect of modeling the high-order connectivity or neighbors.
- NGCF consistently yields the best performance on all the datasets. In particular, NGCF improves over the strongest baselines w.r.t. recall@20 by <sup>11</sup>.68%, <sup>11</sup>.97%, and <sup>9</sup>.61% in Gowalla, Yelp2018<sup>∗</sup> , and Amazon-Book, respectively. By stacking multiple embedding propagation layers, NGCF is capable of exploring the high-order connectivity in an explicit way, while CMN and GC-MC only utilize the first-order neighbors to guide the representation learning. This verifies the importance of capturing collaborative signal in the embedding function. Moreover, compared with PinSage, NGCF considers multi-grained representations to infer user preference, while PinSage only uses the output of the last layer. This demonstrates that different propagation layers encode different information in the representations. And the improvements over HOP-Rec indicate that explicit encoding CF in the embedding function can achieve better representations. We conduct one-sample t-tests and p-value < <sup>0</sup>.<sup>05</sup> indicates that the improvements of NGCF over the strongest baseline (underlined) are statistically significant.

4.3.2 Performance Comparison w.r.t. Interaction Sparsity Levels. The sparsity issue usually limits the expressiveness of recommender systems, since few interactions of inactive users are insufficient to generate high-quality representations. We investigate whether exploiting connectivity information helps to alleviate this issue.

Towards this end, we perform experiments over user groups of different sparsity levels. In particular, based on interaction number per user, we divide the test set into four groups, each of which has the same total interactions. Taking Gowalla dataset as an example, the interaction numbers per user are less than 24, 50, 117, and 1014 respectively. Figure [4](#page-7-2) illustrates the results w.r.t. ndcg@20 on

<span id="page-6-0"></span><sup>4</sup>We train MF from scratch, and use the trained MF embeddings to initialize NeuMF, GC-MC, PinSage, and NGCF to speed up and stabilize the training process. Since we update the implementation of BPR loss, we rerun the experiments.

<span id="page-7-3"></span><span id="page-7-2"></span>![](_page_7_Figure_0.jpeg)

Figure 4: Performance comparison over the sparsity distribution of user groups on different datasets. Wherein, the background histograms indicate the number of users involved in each group, and the lines demonstrate the performance w.r.t. ndcg@20.

<span id="page-7-6"></span>Table 3: Effect of embedding propagation layer numbers (L).

|        | Gowalla        |        | Yelp2018∗ |        | Amazon-Book |        |
|--------|----------------|--------|-----------|--------|-------------|--------|
|        | ndcg<br>recall |        | recall    | ndcg   | recall      | ndcg   |
| NGCF-1 | 0.1556         | 0.1315 | 0.0543    | 0.0442 | 0.0313      | 0.0241 |
| NGCF-2 | 0.1547         | 0.1307 | 0.0566    | 0.0465 | 0.0330      | 0.0254 |
| NGCF-3 | 0.1569         | 0.1327 | 0.0579    | 0.0477 | 0.0337      | 0.0261 |
| NGCF-4 | 0.1570         | 0.1327 | 0.0566    | 0.0461 | 0.0344      | 0.0263 |

different user groups in Gowalla, Yelp2018<sup>∗</sup> , and Amazon-Book; we see a similar trend for performance w.r.t. recall@20 and omit the part due to the space limitation. We find that:

- NGCF and HOP-Rec consistently outperform all other baselines on all user groups. It demonstrates that exploiting high-order connectivity greatly facilitates the representation learning for inactive users, as the collaborative signal can be effectively captured. Hence, it might be promising to solve the sparsity issue in recommender systems, and we leave it in future work.
- Jointly analyzing Figures [4\(a\),](#page-7-3) [4\(b\),](#page-7-4) and [4\(c\),](#page-7-5) we observe that the improvements achieved in the first two groups (e.g., <sup>8</sup>.49% and <sup>7</sup>.79% over the best baselines separately for < <sup>24</sup> and < <sup>50</sup> in Gowalla) are more significant than that of the others (e.g., <sup>1</sup>.29% for < <sup>1014</sup> Gowalla groups). It verifies that the embedding propagation is beneficial to the relatively inactive users.

#### 4.4 Study of NGCF (RQ2)

As the embedding propagation layer plays a pivotal role in NGCF, we investigate its impact on the performance. We start by exploring the influence of layer numbers. We then study how the Laplacian matrix (i.e., discounting factor <sup>p</sup>ui between user <sup>u</sup> and item <sup>i</sup>) affects the performance. Moreover, we analyze the influences of key factors, such as node dropout and message dropout ratios. We also study the training process of NGCF.

4.4.1 Effect of Layer Numbers. To investigate whether NGCF can benefit from multiple embedding propagation layers, we vary the model depth. In particular, we search the layer numbers in the range of {1, <sup>2</sup>, <sup>3</sup>, <sup>4</sup>}. Table [3](#page-7-6) summarizes the experimental results, wherein NGCF-3 indicates the model with three embedding propagation layers, and similar notations for others. Jointly analyzing Tables [2](#page-6-1) and [3,](#page-7-6) we have the following observations:

• Increasing the depth of NGCF substantially enhances the recommendation cases. Clearly, NGCF-2 and NGCF-3 achieve consistent improvement over NGCF-1 across all the board, which considers the first-order neighbors only. We attribute the

<span id="page-7-5"></span><span id="page-7-4"></span>improvement to the effective modeling of CF effect: collaborative user similarity and collaborative signal are carried by the secondand third-order connectivities, respectively.

- When further stacking propagation layer on the top of NGCF-3, we find that NGCF-4 leads to overfitting on Yelp2018<sup>∗</sup> dataset. This might be caused by applying a too deep architecture might introduce noises to the representation learning. The marginal improvements on the other two datasets verifies that conducting three propagation layers is sufficient to capture the CF signal.
- When varying the number of propagation layers, NGCF is consistently superior to other methods across three datasets. It again verifies the effectiveness of NGCF, empirically showing that explicit modeling of high-order connectivity can greatly facilitate the recommendation task.

<span id="page-7-0"></span>4.4.2 Effect of Embedding Propagation Layer and Layer-Aggregation Mechanism. To investigate how the embedding propagation (i.e., graph convolution) layer affects the performance, we consider the variants of NGCF-1 that use different layers. In particular, we remove the representation interaction between a node and its neighbor from the message passing function (cf. Equation [\(3\)](#page-2-1)) and set it as that of PinSage and GC-MC, termed NGCF-1PinSage and NGCF-1GC-MC respectively. Moreover, following SVD++, we obtain one variant based on Equations [\(12\)](#page-4-1), termed NGCF-1SVD++. We show the results in Table [4](#page-8-0) and have the following findings:

- NGCF-1 is consistently superior to all variants. We attribute the improvements to the representation interactions (i.e., e<sup>u</sup> ⊙ e<sup>i</sup> ), which makes messages being propagated dependent on the affinity between ei and eu and functions like the attention mechanism [\[2\]](#page-9-20). Whereas, all variants only take linear transformation into consideration. It hence verifies the rationality and effectiveness of our embedding propagation function.
- In most cases, NGCF-1SVD++ underperforms NGCF-1PinSage and NGCF-1GC-MC. It illustrates the importance of messages passed by the nodes themselves and the nonlinear transformation.
- Jointly analyzing Tables [2](#page-6-1) and [4,](#page-8-0) we find that, when concatenating all layers' outputs together, NGCF-1PinSage and NGCF-1GC-MC achieve better performance than PinSage and GC-MC, respectively. This emphasizes the significance of layer-aggregation mechanism, which is consistent with [\[38\]](#page-9-11).

<span id="page-7-1"></span>4.4.3 Effect of Dropout. Following the prior work [\[29\]](#page-9-16), we employ node dropout and message dropout techniques to prevent NGCF from overfitting. Figure [5](#page-8-1) plots the effect of message dropout

<span id="page-8-1"></span><span id="page-8-0"></span>

|               | Gowalla |        | Yelp2018∗     |        | Amazon-Book     |        |
|---------------|---------|--------|---------------|--------|-----------------|--------|
|               | recall  | ndcg   | recall        | ndcg   | recall          | ndcg   |
| NGCF-1        | 0.1556  | 0.1315 | 0.0543        | 0.0442 | 0.0313          | 0.0241 |
| NGCF-1SVD++   | 0.1517  | 0.1265 | 0.0504        | 0.0414 | 0.0297          | 0.0232 |
| NGCF-1GC-MC   | 0.1523  | 0.1307 | 0.0518        | 0.0420 | 0.0305          | 0.0234 |
| NGCF-1PinSage | 0.1534  | 0.1308 | 0.0516        | 0.0420 | 0.0293          | 0.0231 |
| (a) Gowalla   |         |        | (b) Yelp2018∗ |        | (c) Amazon-Book |        |

Table 4: Effect of graph convolution layers.

<span id="page-8-2"></span>![](_page_8_Figure_2.jpeg)

![](_page_8_Figure_3.jpeg)

Figure 6: Test performance of each epoch of MF and NGCF. ratio <sup>p</sup><sup>1</sup> and node dropout ratio <sup>p</sup><sup>2</sup> against different evaluation protocols on different datasets.

Between the two dropout strategies, node dropout offers better performance. Taking Gowalla as an example, setting <sup>p</sup><sup>2</sup> as <sup>0</sup>.<sup>2</sup> leads to the highest recall@20 of <sup>0</sup>.1514, which is better than that of message dropout <sup>0</sup>.1506. One reason might be that dropping out all outgoing messages from particular users and items makes the representations robust against not only the influence of particular edges, but also the effect of nodes. Hence, node dropout is more effective than message dropout, which is consistent with the findings of prior effort [\[29\]](#page-9-16). We believe this is an interesting finding, which means that node dropout can be an effective strategy to address overfitting of graph neural networks.

4.4.4 Test Performance w.r.t. Epoch. Figure [6](#page-8-2) shows the test performance w.r.t. recall of each epoch of MF and NGCF. Due to the space limitation, we omit the performance w.r.t. ndcg which has the similar trend. We can see that, NGCF exhibits fast convergence than MF on three datasets. It is reasonable since indirectly connected users and items are involved when optimizing the interaction pairs in mini-batch. Such an observation demonstrates the better model capacity of NGCF and the effectiveness of performing embedding propagation in the embedding space.

## 4.5 Effect of High-order Connectivity (RQ3)

In this section, we attempt to understand how the embedding propagation layer facilitates the representation learning in the embedding space. Towards this end, we randomly selected six users from Gowalla dataset, as well as their relevant items. We observe how their representations are influenced w.r.t. the depth of NGCF.

<span id="page-8-3"></span>![](_page_8_Figure_9.jpeg)

<span id="page-8-4"></span>Figure 7: Visualization of the learned t-SNE transformed representations derived from MF and NGCF-3. Each star represents a user from Gowalla dataset, while the points with the same color denote the relevant items. Best view in color.

Figures [7\(a\)](#page-8-3) and [7\(b\)](#page-8-4) show the visualization of the representations derived from MF (i.e., NGCF-0) and NGCF-3, respectively. Note that the items are from the test set, which are not paired with users in the training phase. There are two key observations:

- The connectivities of users and items are well reflected in the embedding space, that is, they are embedded into the near part of the space. In particular, the representations of NGCF-3 exhibit discernible clustering, meaning that the points with the same colors (i.e., the items consumed by the same users) tend to form the clusters.
- Jointly analyzing the same users (e.g., 12201 and 6880) across Figures [7\(a\)](#page-8-3) and [7\(b\),](#page-8-4) we find that, when stacking three embedding propagation layers, the embeddings of their historical items tend to be closer. It qualitatively verifies that the proposed embedding propagation layer is capable of injecting the explicit collaborative signal (via NGCF-3) into the representations.

#### 5 CONCLUSION AND FUTURE WORK

In this work, we explicitly incorporated collaborative signal into the embedding function of model-based CF. We devised a new framework NGCF, which achieves the target by leveraging high-order connectivities in the user-item integration graph. The key of NGCF is the newly proposed embedding propagation layer, based on which we allow the embeddings of users and items interact with each other to harvest the collaborative signal. Extensive experiments on three real-world datasets demonstrate the rationality and effectiveness of injecting the user-item graph structure into the embedding learning process. In future, we will further improve NGCF by incorporating the attention mechanism [\[2\]](#page-9-20) to learn variable weights for neighbors during embedding propagation and for the connectivities of different orders. This will be beneficial to model generalization and interpretability. Moreover, we are interested in exploring the adversarial learning [\[13\]](#page-9-21) on user/item embedding and the graph structure for enhancing the robustness of NGCF.

This work represents an initial attempt to exploit structural knowledge with the message-passing mechanism in model-based CF and opens up new research possibilities. Specifically, there <span id="page-9-2"></span>are many other forms of structural information can be useful for understanding user behaviors, such as the cross features [\[41\]](#page-9-41) in context-aware and semantics-rich recommendation [\[22,](#page-9-42) [27\]](#page-9-43), item knowledge graph [\[32\]](#page-9-10), and social networks [\[34\]](#page-9-25). For example, by integrating item knowledge graph with user-item graph, we can establish knowledge-aware connectivities between users and items, which help unveil user decision-making process in choosing items. We hope the development of NGCF is beneficial to the reasoning of user online behavior towards more effective and interpretable recommendation.

Acknowledgement: This research is part of NExT++ research and also supported by the Thousand Youth Talents Program 2018. NExT++ is supported by the National Research Foundation, Prime Minister's Office, Singapore under its IRC@SG Funding Initiative.

#### REFERENCES

- <span id="page-9-3"></span>[1] Yixin Cao, Xiang Wang, Xiangnan He, Zikun Hu, and Tat-Seng Chua. 2019. Unifying Knowledge Graph Learning and Recommendation: Towards a Better Understanding of User Preferences. In WWW.
- <span id="page-9-20"></span>[2] Jingyuan Chen, Hanwang Zhang, Xiangnan He, Liqiang Nie, Wei Liu, and Tat-Seng Chua. 2017. Attentive Collaborative Filtering: Multimedia Recommendation with Item- and Component-Level Attention. In SIGIR. 335–344.
- <span id="page-9-27"></span>[3] Zhiyong Cheng, Ying Ding, Lei Zhu, and Mohan S. Kankanhalli. 2018. Aspect-Aware Latent Factor Model: Rating Prediction with Ratings and Reviews. In WWW. 639–648.
- <span id="page-9-14"></span>[4] Michaël Defferrard, Xavier Bresson, and Pierre Vandergheynst. 2016. Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering. In NeurIPS. 3837–3845.
- <span id="page-9-1"></span>[5] Travis Ebesu, Bin Shen, and Yi Fang. 2018. Collaborative Memory Network for Recommendation Systems. In SIGIR. 515–524.
- <span id="page-9-40"></span>[6] Xavier Glorot and Yoshua Bengio. 2010. Understanding the difficulty of training deep feedforward neural networks. In AISTATS. 249–256.
- <span id="page-9-34"></span>[7] Marco Gori and Augusto Pucci. 2007. ItemRank: A Random-Walk Based Scoring Algorithm for Recommender Engines. In IJCAI. 2766–2771.
- <span id="page-9-9"></span>[8] William L. Hamilton, Zhitao Ying, and Jure Leskovec. 2017. Inductive Representation Learning on Large Graphs. In NeurIPS. 1025–1035.
- <span id="page-9-38"></span>[9] Ruining He and Julian McAuley. 2016. Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class Collaborative Filtering. In WWW. 507–517.
- <span id="page-9-37"></span>[10] Ruining He and Julian McAuley. 2016. VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback. In AAAI. 144–150.
- <span id="page-9-30"></span>[11] Xiangnan He and Tat-Seng Chua. 2017. Neural Factorization Machines for Sparse Predictive Analytics. In SIGIR. 355–364.
- <span id="page-9-32"></span>[12] Xiangnan He, Ming Gao, Min-Yen Kan, and Dingxian Wang. 2017. BiRank: Towards Ranking on Bipartite Graphs. TKDE 29, 1 (2017), 57–71.
- <span id="page-9-21"></span>[13] Xiangnan He, Zhankui He, Xiaoyu Du, and Tat-Seng Chua. 2018. Adversarial Personalized Ranking for Recommendation. In SIGIR. 355–364.
- <span id="page-9-4"></span>[14] Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu, and Tat-Seng Chua. 2017. Neural Collaborative Filtering. In WWW. 173–182.
- <span id="page-9-29"></span>[15] Cheng-Kang Hsieh, Longqi Yang, Yin Cui, Tsung-Yi Lin, Serge J. Belongie, and Deborah Estrin. 2017. Collaborative Metric Learning. In WWW. 193–201.
- <span id="page-9-12"></span>[16] Santosh Kabbur, Xia Ning, and George Karypis. 2013. FISM: factored item similarity models for top-N recommender systems. In KDD. 659–667.
- <span id="page-9-22"></span>[17] Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In ICLR.
- <span id="page-9-15"></span>[18] Thomas N. Kipf and Max Welling. 2017. Semi-Supervised Classification with Graph Convolutional Networks. In ICLR.

- <span id="page-9-23"></span>[19] Yehuda Koren. 2008. Factorization meets the neighborhood: a multifaceted collaborative filtering model. In KDD. 426–434.
- <span id="page-9-5"></span>[20] Yehuda Koren, Robert M. Bell, and Chris Volinsky. 2009. Matrix Factorization Techniques for Recommender Systems. IEEE Computer 42, 8 (2009), 30–37.
- <span id="page-9-36"></span>[21] Dawen Liang, Laurent Charlin, James McInerney, and David M. Blei. 2016. Modeling User Exposure in Recommendation. In WWW. 951–961.
- <span id="page-9-42"></span>[22] Zhenguang Liu, Zepeng Wang, Luming Zhang, Rajiv Ratn Shah, Yingjie Xia, Yi Yang, and Xuelong Li. 2017. FastShrinkage: Perceptually-aware Retargeting Toward Mobile Platforms. In MM. 501–509.
- <span id="page-9-18"></span>[23] Andrew L Maas, Awni Y Hannun, and Andrew Y Ng. 2013. Rectifier nonlinearities improve neural network acoustic models. In ICML.
- <span id="page-9-33"></span>[24] Athanasios N Nikolakopoulos and George Karypis. 2019. RecWalk: Nearly Uncoupled Random Walks for Top-N Recommendation. (2019). [25] Jiezhong Qiu, Jian Tang, Hao Ma, Yuxiao Dong, Kuansan Wang, and Jie Tang. 2018.
- <span id="page-9-19"></span>DeepInf: Social Influence Prediction with Deep Learning. In KDD. 2110–2119.
- <span id="page-9-8"></span>[26] Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback. In UAI. 452– 461.
- <span id="page-9-43"></span>[27] Xuemeng Song, Fuli Feng, Xianjing Han, Xin Yang, Wei Liu, and Liqiang Nie. 2018. Neural Compatibility Modeling with Attentive Knowledge Distillation. In SIGIR. 5–14.
- <span id="page-9-7"></span>[28] Yi Tay, Luu Anh Tuan, and Siu Cheung Hui. 2018. Latent relational metric learning via memory-based attention for collaborative ranking. In WWW. 729–739.
- <span id="page-9-16"></span>[29] Rianne van den Berg, Thomas N. Kipf, and Max Welling. 2017. Graph Convolutional Matrix Completion. In KDD.
- <span id="page-9-6"></span>[30] Hao Wang, Naiyan Wang, and Dit-Yan Yeung. 2015. Collaborative Deep Learning for Recommender Systems. In KDD. 1235–1244.
- <span id="page-9-39"></span>[31] Jun Wang, Lantao Yu, Weinan Zhang, Yu Gong, Yinghui Xu, Benyou Wang, Peng Zhang, and Dell Zhang. 2017. IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models. In SIGIR. 515–524.
- <span id="page-9-10"></span>[32] Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu, and Tat-Seng Chua. 2019. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD.
- <span id="page-9-24"></span>[33] Xiang Wang, Xiangnan He, Fuli Feng, Liqiang Nie, and Tat-Seng Chua. 2018. TEM: Tree-enhanced Embedding Model for Explainable Recommendation. In WWW. 1543–1552.
- <span id="page-9-25"></span>[34] Xiang Wang, Xiangnan He, Liqiang Nie, and Tat-Seng Chua. 2017. Item Silk Road: Recommending Items from Information Domains to Social Users. In SIGIR. 185–194.
- <span id="page-9-28"></span>[35] Xiang Wang, Dingxian Wang, Canran Xu, Xiangnan He, Yixin Cao, and Tat-Seng Chua. 2019. Explainable Reasoning over Knowledge Graphs for Recommendation. In AAAI.
- <span id="page-9-31"></span>[36] Yao Wu, Christopher DuBois, Alice X. Zheng, and Martin Ester. 2016. Collaborative Denoising Auto-Encoders for Top-N Recommender Systems. In WSDM. 153–162.
- <span id="page-9-26"></span>[37] Xin Xin, Xiangnan He, Yongfeng Zhang, Yongdong Zhang, and Joemon Jose. 2019. Relational Collaborative Filtering:Modeling Multiple Item Relations for Recommendation. In SIGIR.
- <span id="page-9-11"></span>[38] Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie Jegelka. 2018. Representation Learning on Graphs with Jumping Knowledge Networks. In ICML, Vol. 80. 5449–5458.
- <span id="page-9-13"></span>[39] Feng Xue, Xiangnan He, Xiang Wang, Jiandong Xu, Kai Liu, and Richang Hong. 2019. Deep Item-based Collaborative Filtering for Top-N Recommendation. TOIS 37, 3 (2019), 33:1–33:25.
- <span id="page-9-0"></span>[40] Jheng-Hong Yang, Chih-Ming Chen, Chuan-Ju Wang, and Ming-Feng Tsai. 2018. HOP-rec: high-order proximity for implicit recommendation. In RecSys. 140–144.
- <span id="page-9-41"></span>[41] Xun Yang, Xiangnan He, Xiang Wang, Yunshan Ma, Fuli Feng, Meng Wang, and Tat-Seng Chua. 2019. Interpretable Fashion Matching with Rich Attributes. In SIGIR.
- <span id="page-9-17"></span>[42] Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, and Jure Leskovec. 2018. Graph Convolutional Neural Networks for Web-Scale Recommender Systems. In KDD (Data Science track). 974–983.
- <span id="page-9-35"></span>[43] Lei Zheng, Chun-Ta Lu, Fei Jiang, Jiawei Zhang, and Philip S. Yu. 2018. Spectral collaborative filtering. In RecSys. 311–319.