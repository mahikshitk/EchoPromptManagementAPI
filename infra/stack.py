"""Prompt Management stack, laid out like RunManagementStack. It imports, but never creates or modifies,
the VPC, private subnets, database-Lambda security group, AtrionDB secret, OpenRouter secret, and the
RunManagement Cognito user pool and UI clients."""

import os
from dataclasses import dataclass

import aws_cdk as cdk
from aws_cdk import aws_apigatewayv2 as apigw
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_secretsmanager as secretsmanager
from aws_cdk.aws_apigatewayv2_authorizers import HttpJwtAuthorizer
from aws_cdk.aws_apigatewayv2_integrations import HttpLambdaIntegration
from constructs import Construct, Node

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class DeploymentConfiguration:
    environment: str
    account: str
    region: str
    vpc_id: str
    private_subnet_ids: list[str]
    private_subnet_availability_zones: list[str]
    database_lambda_security_group_id: str
    atrion_db_secret_arn: str
    open_router_secret_arn: str
    user_pool_id: str
    interactive_client_id: str
    admin_client_id: str
    cors_allowed_origins: list[str]
    open_router_model: str

    @classmethod
    def from_context(cls, node: Node) -> "DeploymentConfiguration":
        def required(key: str) -> str:
            value = str(node.try_get_context(key) or "").strip()
            if not value:
                raise ValueError(f"CDK context value '{key}' is required.")
            return value

        def items(key: str) -> list[str]:
            return [v.strip() for v in required(key).split(",") if v.strip()]

        return cls(
            environment=required("environment"),
            account=required("account"),
            region=required("region"),
            vpc_id=required("vpcId"),
            private_subnet_ids=items("privateSubnetIds"),
            private_subnet_availability_zones=items("privateSubnetAvailabilityZones"),
            database_lambda_security_group_id=required("databaseLambdaSecurityGroupId"),
            atrion_db_secret_arn=required("atrionDbSecretArn"),
            open_router_secret_arn=required("openRouterSecretArn"),
            user_pool_id=required("userPoolId"),
            interactive_client_id=required("interactiveClientId"),
            admin_client_id=required("adminClientId"),
            # unset allows every origin for now; the API takes bearer tokens, not cookies, so no credentials
            cors_allowed_origins=[v.strip() for v in str(node.try_get_context("corsAllowedOrigins") or "*")
                                  .split(",") if v.strip()] or ["*"],
            open_router_model=str(node.try_get_context("openRouterModel") or "").strip(),
        )


class PromptManagementStack(cdk.Stack):
    def __init__(self, scope: Construct, construct_id: str, config: DeploymentConfiguration, **kwargs):
        super().__init__(scope, construct_id, **kwargs)
        prefix = f"prompt-management-{config.environment}"
        authority = f"https://cognito-idp.{config.region}.amazonaws.com/{config.user_pool_id}"
        ui_clients = [config.interactive_client_id, config.admin_client_id]
        image = ecr_assets.DockerImageAsset(self, "Image", directory=ROOT,
                                            platform=ecr_assets.Platform.LINUX_AMD64)

        def function(id: str, name: str, handler: str, **props) -> lambda_.DockerImageFunction:
            return lambda_.DockerImageFunction(
                self, id, function_name=f"{prefix}-{name}",
                code=lambda_.DockerImageCode.from_ecr(image.repository, tag_or_digest=image.image_tag,
                                                      cmd=[handler]),
                architecture=lambda_.Architecture.X86_64, tracing=lambda_.Tracing.ACTIVE,
                log_group=self._log_group(f"{id}LogGroup", f"{prefix}-{name}"), **props)

        # database Lambda: the only code in the VPC and the only holder of AtrionDB credentials
        if len(config.private_subnet_ids) != len(config.private_subnet_availability_zones):
            raise ValueError("Each private subnet must have one matching availability zone.")
        vpc = ec2.Vpc.from_vpc_attributes(self, "Vpc", vpc_id=config.vpc_id,
                                          availability_zones=config.private_subnet_availability_zones,
                                          private_subnet_ids=config.private_subnet_ids)
        subnets = [ec2.Subnet.from_subnet_id(self, f"PrivateSubnet{i + 1}", s)
                   for i, s in enumerate(config.private_subnet_ids)]
        security_group = ec2.SecurityGroup.from_security_group_id(
            self, "DatabaseLambdaSecurityGroup", config.database_lambda_security_group_id, mutable=False)
        db_secret = secretsmanager.Secret.from_secret_complete_arn(self, "AtrionDbSecret",
                                                                   config.atrion_db_secret_arn)
        database = function(
            "DatabaseFunction", "database", "database.handler",
            timeout=cdk.Duration.seconds(30), memory_size=512, reserved_concurrent_executions=10,
            vpc=vpc, vpc_subnets=ec2.SubnetSelection(subnets=subnets), security_groups=[security_group],
            environment={"ATRION_DB_SECRET_ARN": db_secret.secret_arn})
        db_secret.grant_read(database)

        # worker: outside the VPC, so it reaches OpenRouter and Google without a NAT route; jobs and the
        # harvest store go through the database Lambda
        open_router_secret = secretsmanager.Secret.from_secret_complete_arn(self, "OpenRouterSecret",
                                                                            config.open_router_secret_arn)
        worker_env = {
            "OPENROUTER_SECRET_ARN": open_router_secret.secret_arn,
            "DATABASE_FUNCTION": database.function_name,
        }
        if config.open_router_model:
            worker_env["OPENROUTER_MODEL"] = config.open_router_model
        worker = function(
            "WorkerFunction", "worker", "worker.handler",
            timeout=cdk.Duration.minutes(10), memory_size=1536, reserved_concurrent_executions=5,
            retry_attempts=0,                       # a failed run is marked failed; the user starts a new one
            max_event_age=cdk.Duration.minutes(15),  # matches the API's stale-job cutoff
            environment=worker_env)
        open_router_secret.grant_read(worker)
        database.grant_invoke(worker)

        api_function = function(
            "ApiFunction", "api", "api.handler",
            timeout=cdk.Duration.seconds(30), memory_size=512, reserved_concurrent_executions=20,
            environment={
                "AUTH_AUTHORITY": authority,
                "AUTH_ALLOWED_CLIENT_IDS": ",".join(ui_clients),
                "DATABASE_FUNCTION": database.function_name,
                "WORKER_FUNCTION": worker.function_name,
            })
        database.grant_invoke(api_function)
        worker.grant_invoke(api_function)

        http_api = self._http_api(prefix, authority, ui_clients, api_function, config)
        cdk.CfnOutput(self, "ApiUrl", value=http_api.api_endpoint)
        cdk.CfnOutput(self, "SwaggerUrl", value=f"{http_api.api_endpoint}/docs")

    def _http_api(self, prefix: str, authority: str, ui_clients: list[str], api_function: lambda_.IFunction,
                  config: DeploymentConfiguration) -> apigw.HttpApi:
        integration = HttpLambdaIntegration("ApiIntegration", api_function)
        authorizer = HttpJwtAuthorizer("JwtAuthorizer", authority, authorizer_name=f"{prefix}-jwt",
                                       jwt_audience=ui_clients)
        http_api = apigw.HttpApi(
            self, "HttpApi", api_name=f"{prefix}-api",
            default_integration=integration, default_authorizer=authorizer,
            cors_preflight=apigw.CorsPreflightOptions(
                allow_origins=config.cors_allowed_origins,
                allow_methods=[apigw.CorsHttpMethod.GET, apigw.CorsHttpMethod.POST, apigw.CorsHttpMethod.OPTIONS],
                allow_headers=["authorization", "content-type", "x-correlation-id"],
                expose_headers=["location", "x-correlation-id"],
                max_age=cdk.Duration.hours(1)))
        for path in ("/health", "/docs", "/openapi.json"):
            http_api.add_routes(path=path, methods=[apigw.HttpMethod.GET], integration=integration,
                                authorizer=apigw.HttpNoneAuthorizer())

        access_logs = logs.LogGroup(self, "HttpApiAccessLogs", log_group_name=f"/aws/apigateway/{prefix}",
                                    retention=logs.RetentionDays.ONE_MONTH)
        access_logs.apply_removal_policy(cdk.RemovalPolicy.RETAIN)
        stage = http_api.default_stage.node.default_child
        stage.access_log_settings = apigw.CfnStage.AccessLogSettingsProperty(
            destination_arn=access_logs.log_group_arn,
            format='{"requestId":"$context.requestId","ip":"$context.identity.sourceIp",'
                   '"requestTime":"$context.requestTime","httpMethod":"$context.httpMethod",'
                   '"routeKey":"$context.routeKey","status":"$context.status","protocol":"$context.protocol",'
                   '"responseLength":"$context.responseLength","integrationError":"$context.integrationErrorMessage"}')
        stage.default_route_settings = apigw.CfnStage.RouteSettingsProperty(
            throttling_burst_limit=20, throttling_rate_limit=10)
        return http_api

    def _log_group(self, id: str, function_name: str) -> logs.LogGroup:
        group = logs.LogGroup(self, id, log_group_name=f"/aws/lambda/{function_name}",
                              retention=logs.RetentionDays.ONE_MONTH)
        group.apply_removal_policy(cdk.RemovalPolicy.RETAIN_ON_UPDATE_OR_DELETE)
        return group
