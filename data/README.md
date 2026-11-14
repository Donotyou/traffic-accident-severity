# Data

Put the dataset here as **`Addis_Ababa_city_RTA.csv`** (Addis Ababa road traffic accidents,
~13k rows, 31 columns, target = `Accident_severity`).

* `src/train.py` looks for it in `data/`, then in the current working directory.
* The notebook looks in its own folder first, then in `../data/`.
* Check the dataset's license/terms before committing the CSV to a public repository.
  If you cannot redistribute it, add `data/*.csv` to `.gitignore` and link to the source in the README.
