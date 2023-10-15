FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["./Pipfile", "./Pipfile.lock", "./flask_app_q6.py", "./"]

RUN pipenv install --system --deploy

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "flask_app_q6:app"]