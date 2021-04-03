---
title: "A Complete Machine Learning Project From Scratch: Model V1"
date: "2021-01-20T23:40:37.121"
layout: post
draft: false
path: "/posts/machine-learning-project-model-v1/"
tags:
  - "Data Science"
  - "Machine Learning"
  - "Engineering"
description: "In this third post in a series on how to build a complete machine learning product from scratch, I describe how to build an initial model with an associated training/evaluation pipeline and functionality tests."
---

<figure>
    <img src="./sunrise.jpg" alt="model v1 begins">
</figure>

In this post, we will continue where our [previous post](/posts/performing-exploratory-data-analysis) left off and tackle the next step in building a machine learning project which is making our first fully functional training/evaluation pipeline. 

As a quick refresher, remember that our goal is to apply a data-driven solution to the problem of fake news detection taking it from initial setup through to deployment. The phases we will conduct include the following:

1. [Ideation, organizing your codebase, and setting up tooling](/posts/setting-up-a-machine-learning-project/)
2. [Dataset acquisition and exploratory data analysis](/posts/performing-exploratory-data-analysis)
3. **Building and testing the pipeline with a v1 model (this post!)**
4. [Performing error analysis and iterating toward a v2 model](/posts/machine-learning-project-error-analysis-model-v2/)
5. [Deploying the model and connecting a continuous integration solution](/posts/machine-learning-project-model-deployment/)

This article will focus a lot on the machine learning modelling aspects but will also emphasize how to engineer a good pipeline using appropriate abstractions. Full source code is [here](https://github.com/mihail911/fake-news).

## Data Preprocessing

As a first step in building our model, we need to get data in the right format for ingestion into the training pipeline. Recall that in the [last post](/posts/performing-exploratory-data-analysis), we had done some initial exploratory analysis to understand our dataset's characteristics. 

Now we want to take those data insights and clean and preprocess the initial raw data into the $X \rightarrow y$ mapping we need for supervised learning.

To do this, we will define a number of stand-alone scripts that do this cleaning and preprocessing. 

We use stand-alone scripts because this makes it easier to separate out pieces of functionality that can be eventually be used as executable stages in our pipeline. In practice, these stages are eventually automated as workflows with tools like [Airflow](https://airflow.apache.org/).

We first define a script called *normalize\_and\_clean\_data.py* (full reference [here](https://github.com/mihail911/fake-news/blob/master/scripts/normalize_and_clean_data.py)). This script first reads in our data:

```python
def read_datapoints(datapath: str) -> List[Dict]:
    with open(datapath) as f:
        reader = csv.DictReader(f, delimiter="\t", fieldnames=[
            "id",
            "statement_json",
            "label",
            "statement",
            "subject",
            "speaker",
            "speaker_title",
            "state_info",
            "party_affiliation",
            "barely_true_count",
            "false_count",
            "half_true_count",
            "mostly_true_count",
            "pants_fire_count",
            "context",
            "justification"
        ])
        return [row for row in reader]
```

and then performs the normalization and cleaning: 

```python
def normalize_and_clean(datapoints: List[Dict]) -> List[Dict]:
    return normalize_and_clean_speaker_title(
        normalize_and_clean_party_affiliations(
            normalize_and_clean_state_info(
                normalize_and_clean_counts(
                    normalize_labels(
                        datapoints
                    )
                )
            )
        )
    )
```

Each *normalize\_and\_clean* function cleans one field from the original dataset. As an example, for the **speaker_title** field we have:

```python
def normalize_and_clean_speaker_title(datapoints: List[Dict]) -> List[Dict]:
    normalized_datapoints = []
    for datapoint in datapoints:
        # First do simple cleaning
        normalized_datapoint = deepcopy(datapoint)
        old_speaker_title = normalized_datapoint["speaker_title"]
        old_speaker_title = old_speaker_title.lower().strip().replace("-", " ")
        # Then canonicalize
        if old_speaker_title in CANONICAL_SPEAKER_TITLES:
            old_speaker_title = CANONICAL_SPEAKER_TITLES[old_speaker_title]
        normalized_datapoint["speaker_title"] = old_speaker_title
        normalized_datapoints.append(normalized_datapoint)
    return normalized_datapoints
```

To produce consistent entries, this function lowercases the field, strips off any whitespace, and then replaces certain characters like dashes. 

Afterwards to ensure terms that are semantically consistent but spelled differently (something that happened quite a bit in the original dataset) are mapped identically, we convert them all to a canonical form through a mapping we create by looking through the data. This makes it so that terms like **talks show host** and **talk show host** both map to **talk show host**.

Sometimes during model training, we will want to extract some metrics from our data and use them as additional features. These could be something like historical user activity over the last year for a recommendation system.

To make this concrete, for our use case we discovered during exploratory data analysis that distributions of "credit history" counts could be useful information to provide our model. Because the spread of counts can be pretty wide, we bin the values into 10 bins. 

Because our offline feature extraction could be an expensive operation (it isn't here but let's exercise good engineering practice), we separate out the bin-forming into its own script *compute\_credit\_bins.py* whose main functionality is:

```python
if __name__ == "__main__":
    args = read_args()
    train_df = pd.read_json(args.train_data_path, orient="records")
    optimal_credit_bins = {}
    for credit_name in ["barely_true_count",
                        "false_count",
                        "half_true_count",
                        "mostly_true_count",
                        "pants_fire_count"]:
        optimal_credit_bins[credit_name] = list(np.histogram_bin_edges(train_df[credit_name],
                                                                       bins=10))
    with open(args.output_path, "w") as f:
        print(optimal_credit_bins)
        json.dump(optimal_credit_bins, f)
```

One engineering design decision that's worth noting here: for each of these cleaning and processing steps, we ensure that the scripts operate on input files and produce output files. This guarantees that our inputs remain immutable and our stage is easily reproducible. 

With that, we are ready to get our hands dirty with the model (i.e the "real" machine learning).

## Defining the Model

When defining the model, we certainly have a number of choices around architecture and model class. The exact model we choose is a function of a number of factors such as optimal performance on downstream metrics but also practical considerations like resource contraints or latency.

In the early stages of a project, one of the most important considerations is not to focus too much on getting the best-performing model.

Rather the goal is to get *some* model fully integrated into a pipeline where it is servicing requests to a user, so that you can actually measure the product-level metrics you care about (user engagement, click-through rate, etc.) These metrics will inform any further model development efforts.

So in the interest of getting some model up that performs reasonably well, we will choose a [random forest](https://en.wikipedia.org/wiki/Random_forest) as our baseline model. You can certainly use something even more simplistic (like a manual, rule-based system) if that allows you to get it fully integrated in your downstream application more quickly. 

For our purposes, a random forest is relatively easy to get set up and it is often used as a go-to model class for any new problem. Random forests can be quite competitive and even state-of-the-art if tuned correctly.

But before we build our random forest, let's define a general model interface that we want our random forest (and any other models we build) to adhere to. This is an exercise in good software engineering. 

Here is our base model class definition that all other models should inherit from. The full definition is [here](https://github.com/mihail911/fake-news/blob/master/fake_news/model/base.py).

```python
class Model(ABC):
    @abstractmethod
    def train(self,
              train_datapoints: List[Datapoint],
              val_datapoints: List[Datapoint],
              cache_featurizer: Optional[bool] = False) -> None:
        """
        Performs training of model. The exact train implementations are model specific.
        :param train_datapoints: List of train datapoints
        :param val_datapoints: List of validation datapoints that can be used
        :param cache_featurizer: Whether or not to cache the model featurizer
        :return:
        """
        pass
    
    @abstractmethod
    def predict(self, datapoints: List[Datapoint]) -> np.array:
        """
        Performs inference of model on collection of datapoints. Returns an
        array of model predictions. This should only be called after the model
        has been trained.
        :param datapoints: List of datapoints to perform inference on
        :return: Array of predictions
        """
        pass
    
    @abstractmethod
    def compute_metrics(self, eval_datapoints: List[Datapoint], split: Optional[str] = None) -> Dict:
        """
        Compute a set of model-specifc metrics on the provided set of datapoints.
        :param eval_datapoints: Datapoints to compute metrics for
        :param split: Data split on which metrics are being computed
        :return: A dictionary mapping from the name of the metric to its value
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        """
        Return the model-specific parameters such as number of hidden-units in the case
        of a neural network or number of trees for a random forest
        :return: Dictionary containing the model parameters
        """
        pass

```

Given this model interface combined with the fact that we are leveraging [Scikit-Learn](https://scikit-learn.org/stable/) to power our model, the [actual model definition](https://github.com/mihail911/fake-news/blob/master/fake_news/model/tree_based.py) is not too bad:

```python
class RandomForestModel(Model):
    def __init__(self, config: Optional[Dict] = None):
        self.config = config
        model_cache_path = os.path.join(config["model_output_path"], "model.pkl")
        self.featurizer = TreeFeaturizer(os.path.join(config["featurizer_output_path"],
                                                      "featurizer.pkl"),
                                         config)
        if config["evaluate"] and not os.path.exists(model_cache_path):
            raise ValueError("Model output path does not exist but in `evaluate` mode!")
        if model_cache_path and os.path.exists(model_cache_path):
            LOGGER.info("Loading model from cache...")
            with open(model_cache_path, "rb") as f:
                self.model = pickle.load(f)
        else:
            LOGGER.info("Initializing model from scratch...")
            self.model = RandomForestClassifier(**self.config["params"])
    
    def train(self,
              train_datapoints: List[Datapoint],
              val_datapoints: Optional[List[Datapoint]],
              cache_featurizer: Optional[bool] = False) -> None:
        self.featurizer.fit(train_datapoints)
        if cache_featurizer:
            feature_names = self.featurizer.get_all_feature_names()
            with open(os.path.join(self.config["model_output_path"],
                                   "feature_names.pkl"), "wb") as f:
                pickle.dump(feature_names, f)
            self.featurizer.save(os.path.join(self.config["featurizer_output_path"],
                                              "featurizer.pkl"))
        train_labels = [datapoint.label for datapoint in train_datapoints]
        LOGGER.info("Featurizing data from scratch...")
        train_features = self.featurizer.featurize(train_datapoints)
        self.model.fit(train_features, train_labels)
    
    def compute_metrics(self, eval_datapoints: List[Datapoint], split: Optional[str] = None) -> Dict:
        expected_labels = [datapoint.label for datapoint in eval_datapoints]
        predicted_proba = self.predict(eval_datapoints)
        predicted_labels = np.argmax(predicted_proba, axis=1)
        accuracy = accuracy_score(expected_labels, predicted_labels)
        f1 = f1_score(expected_labels, predicted_labels)
        auc = roc_auc_score(expected_labels, predicted_proba[:, 1])
        conf_mat = confusion_matrix(expected_labels, predicted_labels)
        tn, fp, fn, tp = conf_mat.ravel()
        split_prefix = "" if split is None else split
        return {
            f"{split_prefix} f1": f1,
            f"{split_prefix} accuracy": accuracy,
            f"{split_prefix} auc": auc,
            f"{split_prefix} true negative": tn,
            f"{split_prefix} false negative": fn,
            f"{split_prefix} false positive": fp,
            f"{split_prefix} true positive": tp,
        }
    
    def predict(self, datapoints: List[Datapoint]) -> np.array:
        features = self.featurizer.featurize(datapoints)
        return self.model.predict_proba(features)
    
    def get_params(self) -> Dict:
        return self.model.get_params()
    
    def save(self, model_cache_path: str) -> None:
        LOGGER.info("Saving model to disk...")
        with open(model_cache_path, "wb") as f:
            pickle.dump(self.model, f)

```

While this seems complex, if you look closely the bulk of the actual model training is just a few lines. The remainder is making calls to our featurizer (details below) and performing setup/caching. 

## Features

Machine learning models on their own are useless without a good set of predictive features. 

In our case, we want to give our random forest the right signals to learn adequate fake news discriminators. 

Here we will want to leverage the insights we extracted about the data during our [exploratory data analysis](/posts/performing-exploratory-data-analysis).

We will use two types of features for our initial model: 1) manual features and 2) ngram features.

[Ngram](https://en.wikipedia.org/wiki/N-gram) features are often useful when dealing with text as they allow us to pick up on certain lexical and linguistic patterns in the data. 

We will use [tfidf-weights](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) for these features rather than raw ngrams as these types of weight are often using in information retrieval to help upweight/downweight the importance of certain words. 

A lot of the work done with ngram features is handled for us by libraries (like Scikit-learn). 

We will augment this initial feature set with our own manual features. This category of features is where we can really codify our specific data insights. 

As an example, we choose to extract and [one-hot encode](https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/) various fields such as **speaker**, **speaker_title**, **state_info**, and **party_affiliation**. 

In addition, one particularly interesting field that we saw during our exploratory data analysis was the "credit history." 

These integer-valued counts indicate how many times the given speaker historically made statements that were *barely_true*, *false*, *half-true*, *mostly-true*, or *pants-fire* (completely false). We bin these counts into one of 10 intervals, with each interval's edges defined in a field-specific fashion. 

All this looks like this: 

```python
def extract_manual_features(datapoints: List[Datapoint], optimal_credit_bins: Dict) -> List[Dict]:
    all_features = []
    for datapoint in datapoints:
        features = {}
        features["speaker"] = datapoint.speaker
        features["speaker_title"] = datapoint.speaker_title
        features["state_info"] = datapoint.state_info
        features["party_affiliation"] = datapoint.party_affiliation
        # Compute credit score features
        datapoint = dict(datapoint)
        for feat in ["barely_true_count", "false_count", "half_true_count", "mostly_true_count", "pants_fire_count"]:
            features[feat] = str(compute_bin_idx(datapoint[feat], optimal_credit_bins[feat]))
        all_features.append(features)
    return all_features
```

The heart of the featurization code will look something like this (defined using a bunch of Scikit-learn primitives):

```python
dict_featurizer = DictVectorizer()
tfidf_featurizer = TfidfVectorizer()

statement_transformer = FunctionTransformer(extract_statements)
manual_feature_transformer = FunctionTransformer(partial(extract_manual_features,
                                                            optimal_credit_bins=optimal_credit_bins))

manual_feature_pipeline = Pipeline([
    ("manual_features", manual_feature_transformer),
    ("manual_featurizer", dict_featurizer)
])

ngram_feature_pipeline = Pipeline([
    ("statements", statement_transformer),
    ("ngram_featurizer", tfidf_featurizer)
])

self.combined_featurizer = FeatureUnion([
    ("manual_feature_pipe", manual_feature_pipeline),
    ("ngram_feature_pipe", ngram_feature_pipeline)
])
```

One thing to note above is that once we've extracted separate ngram and manual feature vectors, we concatenate them into a single vector (via the feature union). This aggregated vector now stores all the most salient information we want to capture from our data and provide to our model. 

A final comment about engineering: we will define a separate `Featurizer` class which will expose an interface for training the featurizer (necessary for the ngram-based weights) and featurizing arbitrary data: 

```python
class TreeFeaturizer(object):
    def __init__(*args):
        pass
    
    def fit(self, datapoints: List[Datapoint]) -> None:
        raise NotImplementedError
    
    def featurize(self, datapoints: List[Datapoint]) -> np.array:
        raise NotImplementedError
```

## The Training Pipeline

Ok, so now we have awesome features and a cool model. Let's weave them together into a functional training/evaluation pipeline. 

One important step in any good training pipeline is making it easily configurable, i.e. making it easy to plug-and-play different components/attributes of the pipeline such as model, featurization, or data parameters. 

You typically make these pipelines configurable by either passing in commandline-level arguments or by providing a configuration file. 

I personally prefer the configuration route because it allows the actual configuration file to be version-controlled and tracked in the code repository. 

Additionally, having more than a handful of commandline arguments quickly becomes difficult to deal with. Can you spot the error?

```python
CUDA_VISIBLE_DEVICES=1 python train.py --arg1 val1 --arg3 val3 --arg4 val4 --arg8 val8 --arg2 val2 
--arg11 val11 --arg9 val8 --arg10 val10 --arg14 val14 --arg15 val15 --arg12 val12
```

We will configure our pipeline using a [JSON](https://www.json.org/json-en.html)-formatted file. You can, of course, use other formats like [YAML](https://yaml.org/), etc. but I have a personal bias for JSON because I find it easier to read. 

Another file format that I personally recommend because it's like JSON++ (with support for commenting, templating, etc.) is [Jsonnet](https://jsonnet.org/) though I've stuck with JSON for simplicity-sake. For our random forest model, our config will look something [like this](https://github.com/mihail911/fake-news/blob/master/config/random_forest.json):

```python
{
  "model": "random_forest",
  "train_data_path": "data/processed/cleaned_train_data.json",
  "val_data_path": "data/processed/cleaned_val_data.json",
  "test_data_path": "data/processed/cleaned_test_data.json",
  "featurizer_output_path": "model_checkpoints/random_forest",
  "credit_bins_path": "data/processed/optimal_credit_bins.json",
  "model_output_path": "model_checkpoints/random_forest",
  "evaluate": false,
  "params": {}
}
```

The definition is relatively straightforward. We define our model type, where we should pull data from, where we should write the model and featurizer to, whether we are in evaluate or train mode, and any model-specific parameters. 

We have kept the params empty here because we are using the Scikit-learn random forest defaults but you can use this field to specify anything you want to play with (i.e. number of trees, splitting criterion, etc.)

We next move to our train pipeline, which will be responsible for stitching together everything into an executable series of steps. We've done the bulk of the heavy-lifting in previous sections, so the [pipeline](https://github.com/mihail911/fake-news/blob/master/fake_news/train.py) is largely boilerplate:

```python
if __name__ == "__main__":
    args = read_args()
    with open(args.config_file) as f:
        config = json.load(f)
    
    set_random_seed(42)
    mlflow.set_experiment(config["model"])
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_output_path = os.path.join(base_dir, config["model_output_path"])
    # Update full model output path
    config["model_output_path"] = model_output_path
    os.makedirs(model_output_path, exist_ok=True)
    # Copy config to model directory
    copy(args.config_file, model_output_path)
    with mlflow.start_run() as run:
        with open(os.path.join(model_output_path, "meta.json"), "w") as f:
            json.dump({"mlflow_run_id": run.info.run_id}, f)
        mlflow.set_tags({
            "evaluate": config["evaluate"]
        })
        
        train_data_path = os.path.join(base_dir, config["train_data_path"])
        val_data_path = os.path.join(base_dir, config["val_data_path"])
        test_data_path = os.path.join(base_dir, config["test_data_path"])
        # Read data
        train_datapoints = read_json_data(train_data_path)
        val_datapoints = read_json_data(val_data_path)
        test_datapoints = read_json_data(test_data_path)
        
        if config["model"] == "random_forest":
            config["featurizer_output_path"] = os.path.join(base_dir, config["featurizer_output_path"])
            model = RandomForestModel(config)
        else:
            raise ValueError(f"Invalid model type {config['model']} provided")
        
        if not config["evaluate"]:
            LOGGER.info("Training model...")
            model.train(train_datapoints, val_datapoints, cache_featurizer=True)
            if config["model"] == "random_forest":
                # Cache model weights on disk
                model.save(os.path.join(model_output_path, "model.pkl"))
        
        mlflow.log_params(model.get_params())
        LOGGER.info("Evaluating model...")
        metrics = model.compute_metrics(val_datapoints, split="val")
        LOGGER.info(f"Val metrics: {metrics}")
        mlflow.log_metrics(metrics)
```

While the script seems a tad long, there's really not much going on here: read the config, load up the data, train the model, evaluate the model, and log a bunch of stuff. 

Here we are using the tracking API from [MLFlow](https://www.mlflow.org/docs/latest/tracking.html) which is a super handy way to monitor training parameters, metrics, etc. 

I personally find it more feature-rich and general than [Tensorboard](https://www.tensorflow.org/tensorboard), but use what you're comfortable with.

## Functionality Tests

So now we're written all this stuff. How do we know any of this even works?

We can certainly run the full pipeline (and we will!) but are we really going to have to retrain a model from scratch every time we make a small change to featurization code? 

This quickly becomes impractical, especially if we are working with a team on a shared codebase (i.e. does **everyone** need to retrain a model locally if I make a small featurization change?)

To get around that problem, we need to include functionality tests. This is a very common practice in broader software engineering but sadly not something I've seen a lot of in the machine learning community. 

There are [a](https://www.jeremyjordan.me/testing-ml/) [few](https://eugeneyan.com/writing/testing-ml/) [resources](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765) that do a good job of covering the different kinds of testing in machine learning, so check them out for further details.

For our purposes we will write a few different functionality tests. 

First off we will test our featurization code. This is super important. 

Incorrect featurization means a confused model means a confused human. 

We will test each of our [normalization functions](https://github.com/mihail911/fake-news/blob/master/tests/test_features.py):

```python
def test_compute_bin_idx():
    bins = [0, 4, 10, 12]
    assert compute_bin_idx(0, bins) == 0
    assert compute_bin_idx(3, bins) == 1
    assert compute_bin_idx(4, bins) == 1
    assert compute_bin_idx(12, bins) == 3


def test_normalize_labels():
    datapoints = [
        {"label": "pants-fire", "ignored_field": "blah"},
        {"label": "barely-true"},
        {"label": "false"},
        {"label": "true"},
        {"label": "half-true"},
        {"label": "mostly-true"}
    ]
    
    expected_converted_datapoints = [
        {"label": False, "ignored_field": "blah"},
        {"label": False},
        {"label": False},
        {"label": True},
        {"label": True},
        {"label": True}
    ]
    
    assert normalize_labels(datapoints) == expected_converted_datapoints
...
```

We will also [test our modelling code](https://github.com/mihail911/fake-news/blob/master/tests/test_model.py). Here we will be checking to see that the output of our functions have the appropriate shape, that they are in the correct range (i.e. $\leq 1$ if probabilities), and that we can overfit a small train set:

```python
def test_rf_overfits_small_dataset(config, sample_datapoints):
    model = RandomForestModel(config=config)
    train_labels = [True, False, True]
    
    model.train(sample_datapoints)
    predicted_labels = np.argmax(model.predict(sample_datapoints), axis=1)
    predicted_labels = list(map(lambda x: bool(x), predicted_labels))
    assert predicted_labels == train_labels


def test_rf_correct_predict_shape(config, sample_datapoints):
    model = RandomForestModel(config=config)
    
    model.train(sample_datapoints)
    predicted_labels = np.argmax(model.predict(sample_datapoints), axis=1)
    
    assert predicted_labels.shape[0] == 3


def test_rf_correct_predict_range(config, sample_datapoints):
    model = RandomForestModel(config=config)
    
    model.train(sample_datapoints)
    predicted_probs = model.predict(sample_datapoints)
    
    assert (predicted_probs <= 1).all()
```

One additional kind of test we will employ that is very important is data integrity testing. 

What we mean here is tests that check whether your data source has the expected format, types, range of values, etc. 

If the data doesn't have the format you expect, the rest of your pipeline will be screwed up since data is the start of your training pipeline. 

Data testing becomes especially important when you have a continuous [ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load) pipeline running that is ingesting, processing, and dumping new data periodically. 

While here we are dealing with a static research dataset, we will go through the exercise of writing data integrity tests anyway. 

To do this, we will use the library [Great Expectations](https://greatexpectations.io/). It allows you to specify what you expect of your data in neat snippets of JSON. 

For our purposes, we will check a few things in our data: 
1. All of our data splits have the expected fields (columns) 
2. The statement field is defined (has a length of at least 1)
3. Each of the "credit history" columns is $\geq 0$
4. The labels are booleans 

In Great Expectations syntax that looks something like this:

```python
{
  "data_asset_type": "Dataset",
  "expectation_suite_name": "fake_news_data_suite",
  "expectations": [
    {
      "expectation_type": "expect_table_columns_to_match_set",
      "kwargs": {
        "column_set": [
          "id",
          "statement_json",
          "label",
          "statement",
          "subject",
          "speaker",
          "speaker_title",
          "state_info",
          "party_affiliation",
          "barely_true_count",
          "false_count",
          "half_true_count",
          "mostly_true_count",
          "pants_fire_count",
          "context",
          "justification"
        ]
      },
      "meta": {}
    },
    {
      "expectation_type": "expect_column_values_to_be_in_set",
      "kwargs": {
        "column": "label",
        "value_set": [
          true,
          false
        ]
      },
      "meta": {}
    },
...
```

Given this data suite, we will execute it programmatically with a [Python script](https://github.com/mihail911/fake-news/blob/master/tests/great_expectations/validate_data.py):

```python

context = ge.data_context.DataContext()

datasource_name = "fake_news_data"

train_batch = context.get_batch(
    {"path": f"{os.environ['GE_DIR']}/data/processed/cleaned_train_data.json",
     "datasource": datasource_name},
    "fake_news_data_suite")
val_batch = context.get_batch(
    {"path": f"{os.environ['GE_DIR']}/data/processed/cleaned_val_data.json",
     "datasource": datasource_name},
    "fake_news_data_suite")
test_batch = context.get_batch(
    {"path": f"{os.environ['GE_DIR']}/data/processed/cleaned_test_data.json",
     "datasource": datasource_name},
    "fake_news_data_suite")

results = context.run_validation_operator(
    "action_list_operator",
    assets_to_validate=[train_batch, val_batch, test_batch],
    run_id=str(datetime.now()))

print(results)
if results["success"]:
    print("Test suite passed!")
    exit(0)
else:
    print("Test suite failed!")
    exit(1)
```

Note that Great Expectations also supports running your data suite directly from the commandline, but we are using a Python script we can invoke manually to make our lives a bit easier when we build a continuous integration system (in a later post of the series).

## Putting It All Together

With all this piping in place, we can finally train our system: 

```python
python train.py --config-file config/random_forest.json
```

Our output will look something like this:
```
INFO - 2021-01-21 21:26:49,779 - features.py - Creating featurizer from scratch...
INFO - 2021-01-21 21:26:49,781 - tree_based.py - Initializing model from scratch...
INFO - 2021-01-21 21:26:49,781 - train.py - Training model...
INFO - 2021-01-21 21:26:50,163 - features.py - Saving featurizer to disk...
INFO - 2021-01-21 21:26:50,169 - tree_based.py - Featurizing data from scratch...
INFO - 2021-01-21 21:26:59,360 - tree_based.py - Saving model to disk...
INFO - 2021-01-21 21:26:59,459 - train.py - Evaluating model...
INFO - 2021-01-21 21:26:59,584 - train.py - Val metrics: {'val f1': 0.7587628865979381, 'val accuracy': 0.7266355140186916, 'val auc': 0.8156070164865074, 'val true negative': 381, 'val false negative': 116, 'val false positive': 235, 'val true positive': 552}
```

And if run on the test set:
```
INFO - 2021-01-21 21:28:46,661 - train.py - Test metrics: {'test f1': 0.7828348504551366, 'test accuracy': 0.7396726422447389, 'test auc': 0.8053471940466883, 'test true negative': 347, 'test false negative': 125, 'test false positive': 209, 'test true positive': 602}
```

According to the results in Table 2 of the [original paper](https://www.aclweb.org/anthology/W18-5513.pdf), we outperform all of the models they analyzed. Nice!

And with that, we are done with this super long post. We've come a long way, going from a handful of data insights after our exploratory data analysis to a fully-fledged pipeline that trains a very competitive model on our dataset. 

Taking a step back, remember that our goal was not to build the best model we could. 

Our goal was to get some reasonably good model that we could expose to users to deliver value and initiate a [data flywheel](https://www.cbinsights.com/research/team-blog/data-network-effects/). 

We could certainly eke out more performance on this model by performing hyperparameter tuning, playing with more features etc. But that is left as an exercise to the reader. 

In the next post, we'll look at analyzing our model errors and building a v2 model!
