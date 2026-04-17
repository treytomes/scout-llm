import json
from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    SecretValue,
    Stack,
    aws_iam as iam,
    aws_s3 as s3,
    aws_secretsmanager as secretsmanager,
    aws_resourcegroups as resourcegroups
)
from constructs import Construct

from util.config import Config


class ScoutLlmStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, config: Config, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.data_bucket = s3.Bucket(
            self,
            "scout-llm-data",
            bucket_name="scout-llm-data",
            versioned=True,
            removal_policy=RemovalPolicy.RETAIN,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            lifecycle_rules=[
                s3.LifecycleRule(
                    enabled=True,
                    abort_incomplete_multipart_upload_after=Duration.days(7),
                    noncurrent_version_expiration=Duration.days(90)
                )
            ]
        )

        self.sagemaker_role = iam.Role(
            self,
            "scout-llm-sagemaker-execution-role",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Execution role for Scout LLM SageMaker training jobs."
        )
        self.sagemaker_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonSageMakerFullAccess"
            )
        )
        self.data_bucket.grant_read_write(self.sagemaker_role)

        resourcegroups.CfnGroup(
            self,
            "scout-llm-resource-group",
            name="scout-llm-resources",
            resource_query={
                "Type": "TAG_FILTERS_1_0",
                "Query": json.dumps({
                    "ResourceTypeFilters": ["AWS::AllSupported"],
                    "TagFilters": [
                        {
                            "Key": "project",
                            "Values": [config.project]
                        }
                    ]
                })
            }
        )
        
        CfnOutput(
            self,
            "scout-llm-output-data-bucket-arn",
            value=self.data_bucket.bucket_arn,
            description="ARN of the Scout LLM data backup bucket.",
            export_name="data-bucket-arn",
        )

        CfnOutput(
            self,
            "scout-llm-output-data-bucket-name",
            value=self.data_bucket.bucket_name,
            description="ARN of the Scout LLM data backup bucket.",
            export_name="data-bucket-name",
        )

        CfnOutput(
            self,
            "scout-llm-output-sagemaker-execution-role-arn",
            value=self.sagemaker_role.role_arn,
            description="IAM role used by SageMaker training jobs"
        )

