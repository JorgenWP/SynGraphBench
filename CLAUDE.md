# Project: SynGraphBench

## Methodology Alignment

All implementations must preserve a fair comparison between generative models. The same constraints must apply consistently across every pipeline stage: training the generative model, generating synthetic data, training GNNs on that data, and evaluating performance. Any change that introduces inconsistent conditions between models or between the real-data baseline and synthetic evaluation violates the methodology. Before implementing anything, ask: does this keep the comparison fair?

## Methodology Notebooks

If a code change alters how something described in an existing Methodology/ notebook works (parameters, implementation, pipeline integration), update that notebook. Never create new notebooks in Methodology/ — only update existing ones.