# GitHub Actions deployment

Two workflows:

- [`ci.yml`](../../.github/workflows/ci.yml) runs on every pull request and push to `main`: unit tests, then
  `cdk synth` with placeholder values (no AWS credentials needed).
- [`deploy.yml`](../../.github/workflows/deploy.yml) is started by hand (**Actions > Deploy > Run workflow**)
  for `dev`, `staging` or `production`. It runs the tests, takes temporary AWS credentials through GitHub
  OIDC, builds the Lambda image on the runner, deploys `PromptManagement-<env>`, calls `GET /health` on the
  new API, and uploads `cdk-outputs.json` (the `ApiUrl`) as a workflow artifact. Two deploys of the same
  environment never run at once.

The stack goes to the same account and region as RunManagementAPI and imports its VPC, subnets,
database-Lambda security group, AtrionDB secret and Cognito user pool, so most values below are the same ones
already set in that repository's GitHub environments.

## 1. Before the first deploy

1. A database owner applies [`database/prerequisites/`](../../database/prerequisites/) `001` and `002` and
   grants the AtrionDB secret's application role `SELECT, INSERT, UPDATE` on `public.atrion_prompt_generation`,
   `public.atrion_prompt_harvest_fetch` and `public.atrion_prompt_harvest_result`.
2. Create a Secrets Manager secret whose whole plaintext value is the OpenRouter API key.
3. The account and region must be CDK-bootstrapped. RunManagementAPI already did this for its account; for a
   new one, an administrator runs `npm ci && npx cdk bootstrap aws://AWS_ACCOUNT_ID/AWS_REGION` once.

## 2. GitHub OIDC deployment role

The account already has the `token.actions.githubusercontent.com` OIDC provider if RunManagementAPI deploys
there. Create a role (one per environment, or one statement per environment) that trusts only this repository
and environment:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::AWS_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:atrionai@280709190/EchoPromptManagementAPI@1408136728:environment:dev"
        }
      }
    }
  ]
}
```

The subject must match exactly what GitHub sends. The org's tokens carry immutable IDs (`<org>@<org id>/<repo>@<repo id>`,
as in the other atrionai deployment roles), so a deleted-and-recreated repo or org with the same name cannot
assume the role. 280709190 is `atrionai`'s ID and 1408136728 this repository's
(`https://api.github.com/repos/atrionai/EchoPromptManagementAPI` → `id`). If the org ever reverts to name-only
subjects, the value is `repo:atrionai/EchoPromptManagementAPI:environment:dev`.

Its permissions only need to assume the CDK bootstrap roles (CloudFormation does the actual work through the
execution role):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sts:AssumeRole",
      "Resource": [
        "arn:aws:iam::AWS_ACCOUNT_ID:role/cdk-hnb659fds-deploy-role-AWS_ACCOUNT_ID-AWS_REGION",
        "arn:aws:iam::AWS_ACCOUNT_ID:role/cdk-hnb659fds-file-publishing-role-AWS_ACCOUNT_ID-AWS_REGION",
        "arn:aws:iam::AWS_ACCOUNT_ID:role/cdk-hnb659fds-image-publishing-role-AWS_ACCOUNT_ID-AWS_REGION",
        "arn:aws:iam::AWS_ACCOUNT_ID:role/cdk-hnb659fds-lookup-role-AWS_ACCOUNT_ID-AWS_REGION"
      ]
    }
  ]
}
```

Do not store AWS access keys in GitHub.

## 3. GitHub environments

Create environments `dev`, `staging` and/or `production` (**Settings > Environments**), add required reviewers
to the protected ones, and set these environment variables (none are secrets; the stack only receives ARNs):

| Variable | Example / source |
| --- | --- |
| `AWS_ROLE_ARN` | the role from step 2 |
| `AWS_ACCOUNT_ID` | `123456789012` |
| `AWS_REGION` | `eu-west-2` |
| `VPC_ID` | same as RunManagementAPI |
| `PRIVATE_SUBNET_IDS` | same as RunManagementAPI, e.g. `subnet-aaa,subnet-bbb` |
| `PRIVATE_SUBNET_AVAILABILITY_ZONES` | same as RunManagementAPI, in the same order as the subnets |
| `DATABASE_LAMBDA_SECURITY_GROUP_ID` | same as RunManagementAPI |
| `ATRION_DB_SECRET_ARN` | same as RunManagementAPI (complete ARN, with its six-character suffix) |
| `OPENROUTER_SECRET_ARN` | complete ARN of the secret from step 1.2 |
| `USER_POOL_ID` | RunManagement stack output |
| `INTERACTIVE_CLIENT_ID` | RunManagement stack output |
| `ADMIN_CLIENT_ID` | RunManagement stack output |
| `CORS_ALLOWED_ORIGINS` | optional; leave unset to allow every origin (`*`), or set the UI origin(s), comma-separated, e.g. `https://app.example.com`, to restrict it |
| `OPENROUTER_MODEL` | optional; leave unset for the code's default model |

## 4. Deploy

**Actions > Deploy > Run workflow**, pick the environment. The run summary shows the API URL, which is the UI's
base URL for this API.

## Upgrading CDK

`aws-cdk` in [`package.json`](../../package.json) and `aws-cdk-lib` in
[`infra/requirements.txt`](../../infra/requirements.txt) are pinned together: a newer library can write a cloud
assembly the older CLI cannot read. Bump both, run `npm install`, and let CI's synth confirm.
