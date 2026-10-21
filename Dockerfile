# One image for all three Lambdas; the CDK stack picks the handler per function.
# Built on Linux, so pandas/numpy/psycopg get Linux wheels whatever machine deploys.
FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY api.py worker.py database.py ${LAMBDA_TASK_ROOT}/
COPY promptgen ${LAMBDA_TASK_ROOT}/promptgen

CMD ["api.handler"]
