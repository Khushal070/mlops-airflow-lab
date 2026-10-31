"""
### Wine clustering pipeline

extract_data → validate_data → preprocess → tune_k → train_final → [predict_samples, build_report]

Clusters the Wine dataset with K-Means, picks k by silhouette score, checks the choice
against the true cultivars, and writes an HTML report to working_data/.
Set k range and the validation row minimum with "Trigger DAG w/ config".
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from src.lab import (
    build_report,
    extract_data,
    predict_samples,
    preprocess,
    train_final,
    tune_k,
    validate_data,
)

default_args = {
    "owner": "Khushal Trivedi",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(minutes=10),
}

with DAG(
    dag_id="wine_clustering_pipeline",
    description="K-Means on the Wine dataset with a validation gate, silhouette k selection and an HTML report",
    start_date=datetime(2026, 10, 1),
    schedule=None,  # manual runs only
    catchup=False,
    default_args=default_args,
    render_template_as_native_obj=True,  # "{{ params.k_min }}" arrives as an int, not a string
    params={
        "min_rows": Param(150, type="integer", minimum=1,
                          description="Validation fails if the dataset has fewer rows than this"),
        "k_min": Param(2, type="integer", minimum=2, description="Smallest k to try"),
        "k_max": Param(10, type="integer", minimum=3, maximum=20, description="Largest k to try"),
    },
    tags=["mlops", "clustering", "lab1"],
    doc_md=__doc__,
) as dag:

    run = {"run_id": "{{ run_id }}"}  # each run writes to its own folder

    extract = PythonOperator(task_id="extract_data", python_callable=extract_data, op_kwargs=run)

    validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
        op_kwargs={**run, "min_rows": "{{ params.min_rows }}"},
        retries=0,  # bad data stays bad on a retry, so fail immediately
    )

    prep = PythonOperator(task_id="preprocess", python_callable=preprocess, op_kwargs=run)

    tune = PythonOperator(
        task_id="tune_k",
        python_callable=tune_k,
        op_kwargs={**run, "k_min": "{{ params.k_min }}", "k_max": "{{ params.k_max }}"},
    )

    train = PythonOperator(task_id="train_final", python_callable=train_final, op_kwargs=run)

    predict = PythonOperator(task_id="predict_samples", python_callable=predict_samples, op_kwargs=run)

    report = PythonOperator(task_id="build_report", python_callable=build_report, op_kwargs=run)

    extract >> validate >> prep >> tune >> train >> [predict, report]