# MSc thesis, Autumn 2019
The content in this repository is still work in progress. I am currently working
towards my deadline, November 15th 2019.

## What
The thesis focus on time series classification with uni-dimensional CNNs, in
which the research question is concerned in answering how this compares to
LSTM-based RNNs.

## Why
LSTM have complex gating mechanisms, requiring extensive computational
resources. Architectures like GRU have been proposed as an alternative, although
in recent years, CNNs have proven to be potential competitors to these these
architectures. 

Moreover, in the field of time-series analysis and time-series classification,
comparative studies on neural networks are very limited. Some studies use RNN or
CNN separately. Overall, comparisons of RNN and CNN do exist, but moreoften, use
cases focus on various language tasks in particular. 

The thesis aims to understand how and why CNN can be applicable to time-series
classification through extensive experiments across three use cases and
datasets, and presents a comparison against LSTM.

## How
* Codebase: ~2000 lines of Python-code
* Most important libraries used: **Keras** and **Pandas**
* Use cases in the medical-, energy- and sports domain
