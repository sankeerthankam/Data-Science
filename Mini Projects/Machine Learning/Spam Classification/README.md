# Project for Classifying Texts as SPAM or NOT SPAM

## 1. Data Cleaning

The first block establishes a clean, mathematically sound foundation for the dataset by handling data ingestion, structural cleaning, and label encoding.

### Technical Breakdown

* **Robust Data Ingestion (`encoding="latin-1"`)**
  Most standard CSV text files default to UTF-8 encoding. However, SMS and email datasets frequently contain European characters, mathematical symbols, or curly quotes that cause UTF-8 decoding to crash. Specifying `encoding='latin-1'` (ISO-8859-1) ensures every character maps correctly without runtime exceptions.
* **Dimensionality Reduction & Feature Renaming**
  Raw text exports often create trailing, empty columns (`Unnamed: 2`, `3`, and `4`) due to misaligned delimiters or commas within the messages. Dropping them reduces memory overhead. Renaming the remaining columns from default headers (`v1`, `v2`) to semantic identifiers (`target`, `text`) makes the code highly readable and self-documenting.
* **Preventing Data Leakage (`drop_duplicates`)**
  Duplicate rows typically sneak into text datasets via mass-broadcast spam messages. If duplicates are not removed *before* performing a train-test split, the exact same message could end up in both the training set and the validation/test set (Data Leakage). This artificially inflates validation accuracy because the model simply memorizes repeated text. Setting `keep='first'` ensures only unique instances remain.
* **Ground-Truth Analysis (`value_counts`)**
  This prints the absolute frequency of the classes, revealing a major class imbalance (far more "ham" than "spam"). Checking this early flags that standard accuracy metrics might be deceptive, signaling a heavy reliance on **Precision** and **F1-Score** during evaluation.
* **Categorical to Numerical Target Encoding (`LabelEncoder`)**
  Machine learning loss functions require numeric matrices; text strings like `"ham"` and `"spam"` cannot be used directly as target outputs. The `LabelEncoder` maps these distinct textual categories into discrete integers:
  * `"ham"` $\rightarrow$ `0` (Negative Class)
  * `"spam"` $\rightarrow$ `1` (Positive Class)
  
  Because this task is strictly binary, using `LabelEncoder` outputs a single array of `0`s and `1`s, preparing the target vector perfectly for loss functions like `binary_crossentropy` or log-loss.

### Structural Workflow Status
* **Inputs:** Cleaned down to exactly two columns (`text` and `target`).
* **Rows:** Stripped of redundant, overlapping data points to prevent data leakage.
* **Targets:** Converted into a machine-readable format and assessed for class imbalance.


## 2. Data Processing

The `transform_text` function prepares raw text data so classical machine learning algorithms (like Naive Bayes or Logistic Regression) can extract clean patterns without being distracted by linguistic noise. 


### Technical Breakdown of `transform_text`

1. **Case Normalization & Tokenization**
   * `text.lower()` standardizes all characters to lowercase. Without this, a machine learning model treats `"Free"`, `"FREE"`, and `"free"` as three completely different features, diluting data patterns.
   * `nltk.word_tokenize` breaks continuous strings of text down into individual linguistic units called **tokens** (words, numbers, and punctuation marks).
2. **Alphanumeric Filtering**
   * The `.isalnum()` check discards standalone special characters or lingering punctuation marks (like `@`, `!`, `$`, `%`, `#`). This ensures that only clean words or numeric strings are evaluated in the main text feature set.
3. **Stopwords & Punctuation Removal**
   * **Stopwords Elimination:** Discards frequently occurring words (e.g., `"am"`, `"the"`, `"is"`, `"to"`, `"how"`) that provide grammatical structure but carry little semantic value. Removing them helps the model focus strictly on unique, highly-predictive keywords (like `"winner"`, `"cash"`, or `"urgent"`).
   * **Set Optimization:** Converting `stopwords.words("english")` into a Python `set` optimizes performance. Checking membership in a list takes $O(N)$ time, whereas a set lookup takes $O(1)$ constant time, significantly accelerating execution across thousands of rows.
4. **Stemming (`PorterStemmer`)**
   * Cuts off prefixes or suffixes to reduce a word down to its base or root form (its "stem"). The `PorterStemmer` maps related variations of a word to a single unified token:
     * `"loving"`, `"loves"`, and `"loved"` $\rightarrow$ `"love"`
     * `"giving"`, `"gives"` $\rightarrow$ `"give"`
   * This drastically collapses the vocabulary size (feature space) and groups shared contexts together, helping the model generalize effectively to unseen text.
5. **Recombination**
   * The clean list of tokens is stitched back together using a single blank space as a delimiter (`" ".join(tokens)`). This returns a clean string representing the normalized text, which is the required input format for text vectorizers like `CountVectorizer` or `TfidfVectorizer` down the line.

### <b> Pipeline Application </b>

The `.apply()` method executes this text cleaning pipeline row-by-row over the raw `text` series, saving the final cleaned text format into a new column called `transformed_text`. 

> **Important Note:** This design keeps the original raw text column completely intact. This is crucial because context-aware deep learning models like BERT rely heavily on original casing, punctuation, and full sentence structures, whereas classical bag-of-words models perform best on the heavily stripped down `transformed_text`.

## 3. EDA 

#### <b> Insights </b>

By observing the structural differences between both generated assets:

Spam Messages Word Cloud: Accents transactional, high-urgency, or financial hooks (e.g., prominent displays of words like `"call", "free", "txt", "prize", or "claim")`.

Ham Messages Word Cloud: Accents casual conversation, personal logistics, and normal vocabulary patterns `(e.g., words like "go", "ok", "love", "come", or "ur")`.

#### <b> Class Distribution </b>

Implications: There is a stark class imbalance in spam classification datasets. This distribution tells us that measuring model quality using simple accuracy_score will be deceptive (a dummy model guessing "ham" every time would achieve high accuracy but fail to catch any spam). This flags that we must optimize for Precision and Recall during the training phases.

By isolating the top 10 most common words across both target subsets, distinct 
linguistic profiles emerge for each class. These structural differences provide 
a strong predictive foundation for our classification algorithms.

![img text]((https://github.com/sankeerthankam/Data-Science/blob/f2e9cd86bee1bd269c07dd220691fb9fe46eeec7/Mini%20Projects/Machine%20Learning/Spam%20Classification/img/Class%20Balance.png))


### Class Profile 1: SPAM (Target: 1)

* **Dominant Vocabulary Characteristics:** High concentration of transactional, urgent, and financial hooks. 
  *Key Tokens:* `"call"`, `"free"`, `"txt"`, `"prize"`, `"claim"`.
* **Predictive Modeling Utility:** Yields exceptionally high feature importance weights for Naive Bayes 
  conditional probabilities; serves as strong indicator tokens for malicious 
  intent.

### Class Profile 2: HAM (Target: 0)

* **Dominant Vocabulary Characteristics:** Dominated by personal pronouns, everyday conversational verbs, and 
  colloquial short-hand expressions. 
  *Key Tokens:* `"go"`, `"ok"`, `"love"`, `"come"`, `"ur"`.
* **Predictive Modeling Utility:** Establishes the baseline linguistic pattern of safe communication, 
  which is vital for helping the model reduce False Positive rates.


### Key Analytical Takeaways

1. **Urgency vs. Casual Interaction**
   * **Spam payload markers:** Spam vocabulary is overwhelmingly 
     action-oriented and commercial. Words like `"prize"` and `"claim"` 
     are explicitly used to manufacture synthetic urgency or capitalize 
     on financial interest.
   * **Ham conversational markers:** Legitimate messages represent 
     unstructured, standard human interaction. They consist of situational 
     planning words (`"go"`, `"come"`) or brief acknowledgments (`"ok"`), 
     which rarely appear with high frequency in broadcast spam.

2. **Feature Leverage for Classical Models**
   * Because the vocabularies of Ham and Spam have remarkably low overlap 
     among their top-frequency tiers, classical models like **Multinomial 
     Naive Bayes** will perform exceptionally well. 
   * These distinct keyword signals allow the model to compute highly 
     polarized class conditional probabilities:
     $$\text{P}(\text{word} \mid \text{Spam}) \quad \text{vs.} \quad \text{P}(\text{word} \mid \text{Ham})$$


## 4. This stage of the notebook transitions from exploratory text analysis to 
predictive modeling. It covers text vectorization, cross-validation, 
automated hyperparameter optimization, and a full performance review 
across classical and ensemble classifiers.


#### Model Training

<b> 6.1. TF-IDF Text Vectorization </b>

<b>Term Frequency-Inverse Document Frequency (TF-IDF):</b> Reflects how
important a word is to a document relative to the entire corpus. It penalizes
highly frequent words (like generic terms) while boosting rare, highly
informative words (like `"prize", "claim", or "txt"`).

<b>Dimensionality Control `max_features=3000`:</b> Limits the vocabulary to the top 3,000 words sorted by term frequency. This prevents sparse matrix explosion, reduces memory overhead, and mitigates the "curse of dimensionality."

#### <b> 6.2. Cross-Validation & Optimization Infrastructure </b>

<b> Robust Train-Test Splitting:</b> A 20% validation holdout split isolates
testing data entirely, guaranteeing an unbiased final accuracy calculation.

<b>Stratification Safety (KFold(n_splits=5)):</b> Evaluates the models using
5-fold cross-validation. This ensures every data point is used for training
and validation exactly once, diminishing model evaluation variance.

Automated Scaling Pipeline: Wrapping models inside make_pipeline(MinMaxScaler(), ...)
guarantees that input spaces are bounded strictly between 0 and 1. This is
technically required to satisfy the mathematical boundaries of MultinomialNB,
which crashes if it encounters negative scaled values. 

## 5. Model Performance Evaluation

### <b> 6.3. Model Performance & Evaluation Breakdown </b>

Based on the execution scores from the holdout dataset evaluation loop, the
classifiers achieved the following performance rankings:

#### <b> Tier 1: Top Performers (Scores > 97.5%) </b> 

<b>Logistic Regression (Score: 0.9807):</b> The strongest single classical
model. Because text vector spaces are highly dimensional and linear,
Logistic Regression computes effective boundary separations with a low risk
of overfitting.

<b> Stacking Model (Score: 0.9787):</b> Combines an SVC and a Random Forest
via a Logistic Regression meta-classifier. It maximizes accuracy by leveraging
the diverse architectural predictions of its underlying base estimators.

<b> Multinomial Naive Bayes (Score: 0.9758):</b> A highly efficient, low-overhead
baseline. It performs exceptionally well here because it relies on word
frequency probabilities, matching the clean keyword signals in our spam
corpus.

#### <b> Tier 2: Strong Contenders (Scores 96% - 97.4%) </b> 

<b> Bernoulli Naive Bayes (Score: 0.9729):</b> Operates on binary word
presence/absence. It remains robust but falls slightly behind Multinomial
NB because it ignores term frequency magnitude.

<b> Random Forest (Score: 0.9642):</b> Dependable ensemble technique, though
limited here by a capped estimator count (n_estimators: 10).

#### <b> Tier 3: Underperforming Architectures (Scores < 93%) </b> 

<b> XGBoost (Score: 0.9275) & CatBoost (Score: 0.8975):</b> Gradient boosting
trees struggle compared to linear classifiers in sparse, highly dimensional
text setups due to their reliance on orthogonal, axis-aligned decision
splits.

<b> Gaussian Naive Bayes (Score: 0.8588):</b> Achieves the lowest performance.
It incorrectly assumes that TF-IDF word frequencies follow a standard,
continuous normal Gaussian curve, forcing a poor mathematical fit onto
the sparse text matrix data.

