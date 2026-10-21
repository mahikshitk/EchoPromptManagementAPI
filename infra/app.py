import aws_cdk as cdk

from stack import DeploymentConfiguration, PromptManagementStack

app = cdk.App()
config = DeploymentConfiguration.from_context(app.node)
PromptManagementStack(
    app, f"PromptManagement-{config.environment}", config,
    env=cdk.Environment(account=config.account, region=config.region),
    description="Prompt Management API: prompt generation jobs for the UI, worker, and AtrionDB Lambda")
app.synth()
