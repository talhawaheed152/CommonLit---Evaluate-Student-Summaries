![Header](./commonlit.jpg)
# CommonLit Evaluate Student Summaries Competition

This repository contains the code and analysis for the [CommonLit Evaluate Student Summaries](https://www.kaggle.com/competitions/commonlit-evaluate-student-summaries) Kaggle competition. The goal is to predict two target scores for student-written summaries:

* **content**: measures how well the summary captured the main ideas of the passage.
* **wording**: measures the clarity and cohesion of the student's writing.

The workflow includes data preprocessing, NLP feature extraction, model training, and submission generation.

---

## Directory Structure

```
├── commonlit-fbise-nlp-features-imroze.ipynb  # Main notebook with data pipelines and modeling
├── submission.csv                             # Final CSV submission file
└── README.md                                  # This documentation file
```

## Data Files

* **prompts\_train.csv**: original passages and prompts for training.
* **summaries\_train.csv**: student summaries and true content/wording scores for training.
* **summaries\_test.csv**: unlabeled student summaries for testing.
* **sample\_submission.csv**: format for predictions.
* **summaries\_nlp\_feats.csv**: precomputed summary-level NLP features (merged into training pipeline).

> Note: Place all CSVs under a `data/` directory or adjust paths in the notebook accordingly.

---

## Requirements

* Python 3.8+
* `numpy`
* `pandas`
* `matplotlib`
* `seaborn`
* `nltk`
* `tqdm`
* `sentence-transformers`
* `language-tool-python` (version 2.7.1)
* `xgboost`
* `catboost`

Install dependencies via:

```bash
pip install -r requirements.txt
# or individually:
# pip install numpy pandas matplotlib seaborn nltk tqdm sentence-transformers language-tool-python xgboost catboost
```

---

## Feature Extraction

The notebook defines an `extract_answer_features` function to compute summary-level NLP features:

* **coherence**: sentence ordering consistency (based on SentenceTransformer embeddings).
* **syllable\_complexity**: average syllable count per word.
* **lexical\_complexity**: ratio of rare/difficult words.
* **grammar\_complexity**: parse-tree depth or part-of-speech distribution features.
* **difficult\_words**: count of words not in common vocabulary.
* **spelling\_errors**: number of spelling mistakes detected by LanguageTool.
* **grammar\_errors**: number of grammar errors detected by LanguageTool.
* **num\_sentences**: total sentence count.
* **answer\_len**: total number of words.
* **sentence\_len\_max/min/avg**: statistics on sentence lengths.

These features are extracted for both training and test summaries and stored in a DataFrame for modeling.

---

## Modeling

Two separate predictive pipelines are implemented:

1. **Wording Score Prediction**

   * **Model**: XGBoost Regressor (`reg:squarederror` objective).
   * **Input**: Feature matrix of the extracted NLP features + embeddings from `all-MiniLM-L6-v2` SentenceTransformer.
   * **Evaluation**: Mean Squared Error on a held-out validation split (10% of training data).

2. **Content Score Prediction**

   * **Model**: CatBoost Regressor (`depth=6`, `iterations=500`).
   * **Input**: Only the NLP features (no transformer embeddings).
   * **Evaluation**: Mean Squared Error on a held-out validation split (10% of training data).

### Final Ensemble

The notebook generates two sets of predictions (`content_preds`, `wording_preds`) on the test set and writes them into `submission.csv`.

---

## Usage

1. Clone this repository and navigate to the project directory.

2. Install required dependencies.

3. Update file paths in the notebook if necessary.

4. Launch the notebook in a Jupyter environment:

   ```bash
   jupyter notebook commonlit-fbise-nlp-features-imroze.ipynb
   ```

5. Execute cells sequentially to reproduce data processing, feature extraction, model training, and submission creation.

---

## Submission

The final predictions are saved as `submission.csv` in the following format:

| id  | content | wording |
| --- | ------- | ------- |
| 0   | 0.9834  | 0.5421  |
| ... | ...     | ...     |

Upload `submission.csv` to the Kaggle competition to evaluate your leaderboard score.

---

## Acknowledgments

* [CommonLit](https://commonlit.org/) for providing the dataset.
* Kaggle community for discussions and shared kernels.

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
