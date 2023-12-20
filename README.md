# TrustEdge Bank - Interest Rate Prediction

This "Loan Rate Prediction" repository hosts the cutting-edge predictive model of TrustEdge Bank, designed to estimate loan interest rates. This model leverages machine learning to provide accurate rate predictions, facilitating better loan pricing strategies and financial decision-making. The project stands as a testament to TrustEdge Bank's commitment to technological innovation in finance.

Inaccurately calculated loan interest rates can have significant repercussions for a bank like TrustEdge. Overestimating rates may lead to reduced competitiveness and a loss of potential customers to institutions offering more favorable terms. Underestimating rates can erode profit margins and affect the bank's ability to cover losses from defaults. Both scenarios can impact the bank's reputation, financial stability, and trust with stakeholders, underscoring the crucial nature of precise interest rate prediction in the banking industry.

The model has an **RMSE of 4%.**

This project is part of the Capstone Project for [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp)




## Run Locally

Pre-requisites

```bash
python
git
docker
pandas
numpy
streamlit
```
Activate docker desktop

Clone the project

```bash
  git clone https://github.com/Haroldgio28/Loan_rate_prediction.git
```

Go to the project directory

```bash
  cd my-project
```

Build the docker image

```bash
  docker build --no-cache  -t loan . 
```

Run the application using Docker

```bash
  docker run -it --rm -p 9696:9696 loan
```

Run streamlit app from other command line and open, a window will open on web browser:

```bash
  streamlit run streamlit.py
```

For make an inference of interest rate, there are two ways:

1. Running scripts or notebook from Visual Studio Code:
- Open the [notebook](https://github.com/Haroldgio28/Loan_rate_prediction/blob/main/notebooks/predict-test.ipynb) on Visual Studio Code or other code editor, and change the parameters.
- In another command line, execute the [script](https://github.com/Haroldgio28/Airline-delay-prediction/blob/main/predict-test.py), changing the parameters on the script.

2. Modify the [file](https://github.com/Haroldgio28/Loan_rate_prediction/blob/main/data/customer.csv) and upload to streamlit app 
   ![](https://github.com/Haroldgio28/Loan_rate_prediction/blob/main/app/test_app.gif)

## Next Steps

- Improve the RMSE metric

## Appendix

### Above the data
The dataset used in this project comprises 10,000 records of of loans made through the Lending Club platform, which is a platform that allows individuals to lend to other individuals. It has been sourced from [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/lending-club).


## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/Haroldgio28)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/haroldgiovannyuribe/)
[![twitter](https://img.shields.io/badge/twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)](https://twitter.com/HaroldGio28)
