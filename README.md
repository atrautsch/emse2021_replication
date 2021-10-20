# On the differences between quality increasing and other changes in open source Java projects

The static source code metrics were collected previously as part of the [SmartSHARK](https://www.github.com/smartshark/) ecosystem.
To recreate the collection process a combination of the plugins [vcsSHARK](https://www.github.com/smartshark/vcsSHARK), [mecoSHARK](https://www.github.com/smartshark/mecoSHARK) and optionally [serverSHARK](https://www.github.com/smartshark/serverSHARK) for orchestration of plugins is needed.

We provide several different entry points for this replication kit.
The [first](#raw-data) starts with a database dump as the source and extracts the data from the database.
The [second](#to-re-create-the-plots-and-tables-from-the-paper) starts with the already extracted data and just re-created the plots and tables in the paper.
In addition, we provide our fine-tuned model to play with live on the [website](https://user.informatik.uni-goettingen.de/~trautsch2/emse_2021/).

## Raw data

The raw data used in this paper comes from a SmartSHARK database dump containing 54 Apache projects in Java used in a previous study. If you only want to re-create the plots and tables you can skip this section.


### Restore MongoDB dump
Download the database dump and import it into your local MongoDB. You may have to create a user first.
While we have a dump that only contains our study subjects, the data is also contained in any official release > 2.1 of [SmartSHARK](https://smartshark.github.io/dbreleases).

```bash
wget https://mediocre.hosting/smartshark_emse.agz

# restore mongodb dump
mongorestore -uUSER -p'PASSWORD' --authenticationDatabase=admin --gzip --archive=smartshark_emse.agz
```

### Extract changes from MongoDB

After restoring the MongoDB dump, the changes need to be extracted.

```bash
python diff_metrics.py
```

This creates pickle files, one for each project, this is not very space efficient. It creates about ~4GB of data which is also the reason we do not bundle this.


### Compute manual classification and merge with metric diffs

The raw data from the manual classification phase is available in './data/change\_type\_label\_export2.pickle'.
This is an export from the [visualSHARK](https://github.com/smartshark/visualshark) frontend.

This is further aggregated in the Jupyter Notebook 'notebooks/ReadManualData.ipynb'.
The results of this step and the pickled files from 'diff\_metrics.py' are then aggregated in the Jupyter Notebook 'notebooks/CreateDataset.ipynb'.

### Add predictions for the rest of the data

The metrics data with available ground truth manual classifications is now available in 'data/only\_changes.csv' and 'data\all\_changes.csv'. To complement the ground truth with predictions for all other data we first fetch the fine-tuned model. The fine-tuning step itself is explained [in the model evaluation part.](#evaluation-of-pre-trained-sebert-for-commit-intent-classification)

```bash
cd ft/fine_tuned
wget https://smartshark2.informatik.uni-goettingen.de/sebert/seBERT_fine_tuned_commit_intent.tar.gz
tar -xzf seBERT_fine_tuned_commit_intent.tar.gz
```

Then we add the predictions of the fine-tuned model via the 'AddPredictrions.ipynb' notebook.


## To re-create the plots and tables from the paper

This uses all of the aggregated data to create the plots and tables.

### Create virtualenv and install dependencies
```bash
python -m venv .
source bin/activate
pip install -r requirements.txt
```

### Load data

Due to the restriction of Github for files > 100 MB we do not bundle these files.
```bash
cd data
wget https://zenodo.org/record/5494134/files/all_changes_sebert.csv.gz
```

### Run Jupyter notebooks

The final data is distributed in the data directory.
For all plots and tables only 'notebooks/CreatePlotsTables.ipynb' is needed.

```bash
source bin/activate
cd notebooks
jupyter lab
```

## Evaluation of pre-trained seBERT for commit intent classification

To evaluate if the pre-trained seBERT can be adapted for this task with fine-tuning we run 100 fine-tuning iterations and measure the classification performance.


### Load pre-trained model

```bash
cd ft/models
wget https://smartshark2.informatik.uni-goettingen.de/sebert/seBERT_pre_trained.tar.gz
tar -xzf seBERT_pre_trained.tar.gz
```

### Prepare data

This generates data for each training and testing step.
As we distribute the model evaluation, we use this to make sure that every model uses the same data for each run.

```bash
cd ft
python generate_multi_label_folds.py
```

### Generate HPC scripts

This generates submission scripts for a SLURM HPC system.
It simply generates a submission script for each of the 100 runs for seBERT fine-tuning and the RandomForest baseline.
After generating the scripts they can be submitted to the HPC system.
Be aware that this is using SLURM and that you need about ~900GB of disk space as quite a bit of model data is generated.
The HPC System also needs GPU nodes, in our case we evaluated the model on Nvidia Quattro RTX 5000 GPU nodes.

```bash
cd ft
python generate_commit_intent_ml_scripts.py
```

### Alternative: Jupyter Notebook

We also provide a Jupyter notebook which shows the fine-tuning and evaluation step 'notebooks/FineTuneModel.ipynb'.


### Generate fine-tuned model

After we evaluate the model performance and we are satisfied we generate the final model using all available ground truth data.

```bash
cd ft
python generate_sebert_intent_model.py
```

## Fine-tuned model for commit intent classification

If you just want to try out the final fine-tuned model you can use the live version on the [website](https://user.informatik.uni-goettingen.de/~trautsch2/emse_2021/) or you can use our provided Jupyter Notebook 'notebooks/AddPredictions.ipynb'.
However, you need to download and extract the fine-tuned model first to use the notebook.

```bash
cd ft/fine_tuned
wget https://smartshark2.informatik.uni-goettingen.de/sebert/seBERT_fine_tuned_commit_intent.tar.gz
tar -xzf seBERT_fine_tuned_commit_intent.tar.gz
```



