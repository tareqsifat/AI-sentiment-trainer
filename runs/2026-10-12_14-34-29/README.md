# Training run

Suggested reading order:

1. **00_config.json** – the settings.
2. **01_dataset_stats.json** – how many reviews, label balance, review lengths.
3. **03_tfidf_examples.json** – three reviews walked from raw text to the exact numbers the input layer sees.
4. **02_vocabulary.json** – every input neuron: its token, document frequency, positive share and IDF.
5. **04_architecture.json** – layers, links and parameter count.
6. **epoch_0/** – the random, untrained network. Test accuracy ≈ 50%.
7. **epoch_1/** and later – the same files after each pass over the training data:

| file | what it shows |
|---|---|
| summary.json | loss/accuracy for the epoch, change vs the previous one |
| training_curve.json | running loss/accuracy every N reviews during the epoch |
| sample_trace.json | ONE training step on one review: forward sums, loss, every delta, weight updates, prediction before/after |
| layer_*.json | every weight and bias of each layer |
| neuron_report.json | what each hidden neuron reacts to (positive vs negative reviews, strongest words) |
| word_influence.json | each word fed alone into the network: how positive/negative it reads |
| weight_changes.json | how far the weights moved this epoch, biggest single changes |
| evaluation.json | test-set accuracy, precision, recall, confusion matrix, most confident mistakes |
| checkpoint.json | the full network (weights + biases + vocabulary + IDF), loadable |

Good comparisons:
- `epoch_0/word_influence.json` vs `epoch_1/word_influence.json`: words go from meaningless to meaningful.
- `sample_trace.json` across epochs: the same review, an increasingly confident prediction.
- `neuron_report.json` across epochs: hidden neurons turning into positive/negative detectors.
- `summary.json` train vs test accuracy: when they drift apart the network has started memorizing.
