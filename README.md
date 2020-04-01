# Prune from Scratch
Unofficial implementation of the paper ["Pruning from Scratch"](https://arxiv.org/abs/1909.12579).

In order to verify the validity of the thesis proposed in this paper, I implemented a simple version myself.


## Accuracy
| Model             |Prune ratio        | Acc.        |
| ----------------- | ----------------- |-------------|
| [VGG19](https://arxiv.org/abs/1409.1556)  | 50%          | 93.24%      |

## Insight
At first thought, I think "Pruning from scratch" doesn't make sense. Pruning the network architecture according to the initial random weights doesn't sound reasonable. However, the experiment results show that you did can prune a network from scratch. So I think the key point of "Pruning from scratch" is that the “winning-tickets” subnetwork (LTHLottery Ticket Hypothesis) of the over parameterized network already has better-than-random performance on the data, without any training. Specifically, when we prune a network N from the scratch based on the random weights, we are looking for the "winning tickets" of N actually.  The "Supermask" of the paper "Deconstructing Lottery Tickets: Zeros, Signs, and the Supermask" do the similar things(https://arxiv.org/abs/1905.01067).
