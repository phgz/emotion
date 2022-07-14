# Emotion

A model for emotion classification based on text and audio.

[![emotion - merge](https://github.com/philipGaudreau/emotion/actions/workflows/merge.yml/badge.svg)](https://github.com/philipGaudreau/emotion/actions/workflows/merge.yml)
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)

## Acknowledgements

 - Hafed Benteftifa
 - Soumaya Chaffar

## Features

Give audio and text as input and get back the dominant emotion.

## Usage/Examples

Be sure to add double quotes to lines containing one or more commas for `.csv` files.

For graphical interface, use https://a62-emotion.herokuapp.com/index and upload `.wav` and `.csv` files.

```python
In [1]: import requests

In [2]: with open("./data/raw_sample/audio/UlTJmndbGHM.wav", "rb") as fd:
   ...:     response = requests.post("https://a62-emotion.herokuapp.com/predict", files={'files[]': fd})
   ...:

In [3]: response.json()
Out[3]: {'predictions_audio': {'UlTJmndbGHM.wav': 'positive'}, 'predictions_texte': {}}
```

```bash
curl -F 'files[]=@/home/user/emotion/data/raw_sample/audio/DatH-ra0VKY.wav' -F 'files[]=@/home/user/file.csv' https://a62-emotion.herokuapp.com/predict
{"predictions_audio":{"DatH-ra0VKY.wav":"positive"},"predictions_texte":{"file.csv":{"DAMON WILLIAMS: I am the Senior Vice President and Chief Education and Youth Development Officer for the Boys and Girls Clubs of America.":"neutral","In that role, I help to shape the education policy and youth development direction for more than 4,000 clubs that are part of our federated network, having an impact on more than 4 million young people across this nation.":"positive"}}}
```

## API Reference

#### Get a prediction for file(s)

```http
  POST /predict
```

| Parameter | Type     | Description                      |
| :-------- | :------- | :------------------------------- |
| `files[]` | `File`   | **Required**. File(s) to process |

## Installation

Install emotion with pip

```bash
  pip install a62-emotion
```

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`GDRIVE_CREDENTIALS_DATA`

`FLASK_SECRET_KEY`

## System Dependencies

For audio, you will need to install `libsndfile1` (and `libsndfile1-dev` on some systems) 
```bash
sudo apt-get install libsndfile1
```
## Run Locally

Be sure to have python 3.8.13 as the python executable
```bash
python3 --version
```

To install Poetry, run:

```bash
curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.1.14 python3 - --yes
```

(Optional) To install the Heroku CLI, follow these steps

https://devcenter.heroku.com/articles/heroku-cli#standalone-installation-with-a-tarball

Clone the project

```bash
git clone https://github.com/philipgaudreau/emotion
```

Go to the project directory

```bash
cd emotion
```

Install dependencies (add flag `--without-dev` if you do not want development dependencies)

```bash
poetry install
```

Activate the virtual environment

```bash
poetry shell
```

Pull cloud data if needed (You will need to export `GDRIVE_CREDENTIALS_DATA`)

```bash
dvc pull [<.dvc file>]
```

Start using the command line interface (Not implemented yet)

```bash
emotion --help
```

## Running Tests

To run tests, run the following command (development dependencies must be installed)

```bash
pytest tests
```

## Deployment

This project is automatically deployed on Heroku when a PR is merged. To test locally, run (You will need to export `FLASK_SECRET_KEY`)

```bash
heroku local
```

## Tech Stack

**CI:** GitHub Actions, DVC

**CD:** GitHub Actions, Heroku

**Frameworks:** Scikit-learn, Tensorflow, Flask

**Versionning:** Poetry, SemVer

**Cloud:** G-Drive, GitHub, Heroku

## Feedback

If you have any feedback, please reach out to one of us.


## Authors

- [@philipgaudreau](https://github.com/philipgaudreau)
- [@guraymo](https://github.com/guraymo)
- [@gtrottier](https://github.com/gtrottier)


## 🚀 About Us
We are on our way to finish a degree in Machine Learning.


## License

[MIT](https://choosealicense.com/licenses/mit/)

