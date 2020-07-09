# CA – Assignment 2: Argument Mining

- Group: `FakeNews`
- Group members:

  - Adnan Manzoor
  - Sajjad Pervaiz
  - Kevin Taylor
  - Christoph Schäfer

## Structure

```
.
└── argument-generation-assignment
    ├── Documentation.pdf
    ├── README
    ├── data
    │    ├── essay_prompt_corpus.json
    │    ├── sample_predictions.json
    │    └── train-test-split.csv
    └── code
        ├── TestRank_EssaySummarization.py
        └── model.py
```

### Scripts

`TestRank_EssaySummarization.py`: The ML model that we use for generating predictions.
`evaluation.py`: Script to evaluate the `Rouge F` score.

### How to run the scripts

- Make sure you have the same directory structure as above otherwise adjust the paths in the scripts accordingly.
- Run `TestRank_EssaySummarization.py` to generate the predictions in `data/` directory with name `predictions.json`
- Run `evaluation` script with `predictions.json` as predictions.

## Model Explanation

We started to use a random selection of sentences, to figure out what the lowest baseline is, with results around `0.12 rouge-1 f score`.
Based on that we realised, that to reach the given baseline marginally a not too complex method of sentence selection should suffice.

We followed up by picking sentences by basic features we read about, that indicate a good summarising sentence. These included the longest sentence and selecting sentences based on preselected keywords (e.g. In conclusion..., The best...). This However was not successful enough, as it was in range of the random selection results.

Finally we tried extractive summarisation using the PageRank style evaluation on text, inspired by the given paper from Alshomary et.al.[2].
This lead to a `rouge-1 f score of 0.138`, with which we reached our goal.

Additionally we tested Google's T5 (Text-To-Text Transfer Transformer)[1], which easily produced results higher than the baseline, but we just plugged in the library which didn't seem adequate for the course.  

[1] https://arxiv.org/pdf/1910.10683.pdf

[2] https://webis.de/downloads/publications/papers/alshomary_2020b.pdf