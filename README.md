# Sentiment Trainer

A sentiment classifier written in plain Node.js. Every neuron and every link is its own object, and every epoch writes what the network learned to JSON files.

## Run

Needs Node.js 18 or newer. No `npm install`.

```bash
node server.js
# open http://localhost:3000
```

1. **Upload** the IMDB CSV. Use the Kaggle file "IMDB Dataset of 50K Movie Reviews" (`IMDB Dataset.csv`, columns `review,sentiment`).
2. **Train**. Set *Rows to use* to 5000 for a quick first run, then 0 for the full 50k.
3. Open the log files from section 4, or straight from the `runs/` folder.
4. **Try a sentence** against any epoch, including `epoch_0` (the untrained network).

Command line, without the UI:

```bash
node trainer.js "IMDB Dataset.csv" 5          # 5 epochs, all rows
node trainer.js "IMDB Dataset.csv" 3 5000     # 3 epochs, 5000 rows
```

## Files

| file | what it does |
|---|---|
| `nn.js` | Neuron, Link, Layer, Network: forward, backward, update, save/load |
| `text.js` | cleaning, tokenizing, stopwords, negation, vocabulary, TF-IDF |
| `dataset.js` | CSV parser (handles quotes, commas, newlines inside reviews) |
| `trainer.js` | the training loop and every log file |
| `server.js` | small HTTP server: upload, train in a worker thread, live progress, predict |
| `public/index.html` | the UI |

## Output

```
runs/2026-10-12_10-56-00/
  README.md                 reading guide for this run
  00_config.json
  01_dataset_stats.json
  02_vocabulary.json        each input neuron: token, df, positive share, idf
  03_tfidf_examples.json    3 reviews walked from raw text to input values
  04_architecture.json
  epoch_0/                  random network, before training
  epoch_1/ ...              summary, training_curve, sample_trace, layer_*, neuron_report,
                            word_influence, weight_changes, evaluation, checkpoint
  final_model.json
  99_run_summary.json
```

## Expected numbers on real IMDB

Defaults (5000 vocab, hidden 16,8, lr 0.3): roughly 85–88% test accuracy. Each epoch over 40k reviews takes a few seconds, and each epoch folder is about 5 MB.

The first several thousand steps usually show a flat loss around 0.69 before it drops. That plateau is normal for stacked sigmoid layers.
# AI-sentiment-trainer
