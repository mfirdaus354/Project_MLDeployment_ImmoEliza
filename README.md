# Project_MLDeployment_ImmoEliza - Machine Learning Deployment

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project is a continuation of multiple iterations of ImmoEliza projects, whose objective is to approach the challenge of real estate property valuation by using data science methodology. The end goal of this project is to predict property prices based on various features like plot area, habitable surface, land. This README provides an overview of the project and explains how to use the code.

You can always consult to other notebooks of that is contained in this repository or check other repositories that I have done under ImmoEliza project

 1. [Project Immobel - Exploratory Data Analysis on Belgium's Real Estate Market ](https://github.com/mfirdaus354/project-immobel)
 2. [Project Immo_Regression - My first attempt on generating supervised machine learning model using the data obtained through Project Immobel](https://github.com/mfirdaus354/project_immo_regression)

The ImmoEliza project involves building and deploying a machine learning model to predict property prices. The project includes the following components:

- Data Preprocessing: Data is fetched from an API and preprocessed to handle missing values and scale the features.
- Model Training: The preprocessed data is used to train an XGBoost regression model to predict property prices.
- Model Deployment: The trained model is saved and deployed as a web service using FastAPI.
- API Integration: The frontend can interact with the deployed model through API calls.

## Installation

To set up the project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/ImmoEliza.git`
2. Change into the project directory: `cd ImmoEliza`
3. Install the required dependencies: `pip install -r requirements.txt`

## Usage

To use the project, follow these steps:

1. Ensure the dependencies are installed (see [Installation](#installation)).
2. Run run_uvicorn.py in a Terminal window
        python -m run_uvicorn.py
3. Then, run main.py to generate price prediction. Execute this following command in anoter Teminal Window
        python -m main.py
4. The API will be accessible at `http://127.0.0.1:8000`.

The main files for data preprocessing, model training, and API endpoints are located in the `immo_eliza_api` directory.

## Project Structure

The project follows the following directory structure:

.
└── ./Project_MLDevelopment_ImmoEliza/
    ├── ./Project_MLDevelopment_ImmoEliza/data/
    │   ├── ./Project_MLDevelopment_ImmoEliza/data/categoricals.csv
    │   ├── ./Project_MLDevelopment_ImmoEliza/data/continuous.csv
    │   ├── ./Project_MLDevelopment_ImmoEliza/data/data_forsale_new.csv
    │   ├── ./Project_MLDevelopment_ImmoEliza/data/dataset.json
    │   ├── ./Project_MLDevelopment_ImmoEliza/data/df_sale_clean.csv
    │   └── ./Project_MLDevelopment_ImmoEliza/data/final-csv.csv
    ├── ./Project_MLDevelopment_ImmoEliza/immo_eliza_api/
    │   ├── ./Project_MLDevelopment_ImmoEliza/immo_eliza_api/app.py
    │   ├── ./Project_MLDevelopment_ImmoEliza/immo_eliza_api/crud.py
    │   ├── ./Project_MLDevelopment_ImmoEliza/immo_eliza_api/database.py
    │   ├── ./Project_MLDevelopment_ImmoEliza/immo_eliza_api/models.py
    │   └── ./Project_MLDevelopment_ImmoEliza/immo_eliza_api/schemas.py
    ├── ./Project_MLDevelopment_ImmoEliza/model-building/
    │   ├── ./Project_MLDevelopment_ImmoEliza/model-building/immo_reg_model.ipynb
    │   └── ./Project_MLDevelopment_ImmoEliza/model-building/model_selection_regression.ipynb
    ├── ./Project_MLDevelopment_ImmoEliza/models/
    │   └── ./Project_MLDevelopment_ImmoEliza/models/xgb_reg_model.pkl
    ├── ./Project_MLDevelopment_ImmoEliza/notebooks/
    │   ├── ./Project_MLDevelopment_ImmoEliza/notebooks/immobel_analysis.ipynb
    │   └── ./Project_MLDevelopment_ImmoEliza/notebooks/immobel_insight.ipynb
    ├── ./Project_MLDevelopment_ImmoEliza/output/
    │   ├── ./Project_MLDevelopment_ImmoEliza/output/Parameters.pdf
    │   └── ./Project_MLDevelopment_ImmoEliza/output/price_dist_provinces.png
    ├── ./Project_MLDevelopment_ImmoEliza/src/
    │   ├── ./Project_MLDevelopment_ImmoEliza/src/config.py
    │   └── ./Project_MLDevelopment_ImmoEliza/src/pimp_my_data.py
    ├── ./Project_MLDevelopment_ImmoEliza/virtualenv
    ├── ./Project_MLDevelopment_ImmoEliza/main.py
    ├── ./Project_MLDevelopment_ImmoEliza/prediction.py
    ├── ./Project_MLDevelopment_ImmoEliza/preprocessing.py
    ├── ./Project_MLDevelopment_ImmoEliza/README.md
    ├── ./Project_MLDevelopment_ImmoEliza/requirements.txt
    └── ./Project_MLDevelopment_ImmoEliza/sql_app.db

- `immo_eliza_api`: Contains the FastAPI application and relevant code.
- `notebooks`: Contains Jupyter notebooks for data exploration.
- `model-building`: Contains Jupyter notebooks for supervised machine learning model buildings.
- `models`: Saved model files are stored here.
- `requirements.txt`: Lists the required Python packages for the project.

## Contributing

Contributions to the project are welcome! If you find any bugs or want to add new features, please create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

