FROM apache/airflow:2.10.5

COPY requirements.txt /requirements.txt

# Airflow's constraints file pins every package already in the image (numpy, pandas, ...),
# so installing our packages can't swap them for incompatible versions.
RUN PYTHON_VERSION="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" && \
    pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"