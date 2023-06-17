FROM python:3.10

EXPOSE 8050

WORKDIR /dashboard

COPY . /dashboard/

RUN pip install -r /dashboard/requirements.txt

CMD python bank_churn_dashboard.py