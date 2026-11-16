#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  Create, resize or terminate the ICDE EMR cluster from a machine with the AWS CLI. Creating starts billing:
#  only with Bilal's OK on the estimate (plan doc, "AWS batch").
#
#    KEY=<ec2 key pair> SUBNET=<subnet id> scripts/emr_cluster.sh create [workers]   -> prints the cluster id
#    scripts/emr_cluster.sh step <cluster id> <smoke|expA|expB-read|expB-train|nodes> [N]
#    scripts/emr_cluster.sh resize <cluster id> <workers>
#    scripts/emr_cluster.sh terminate <cluster id>
#
#  AWS validation (advisor 2026-11-11): 1 m5.xlarge primary + 4 m5.2xlarge core nodes, the code and Python 3.11 from
#  a bootstrap action (experiments/aws/README.md):
#    MASTER_TYPE=m5.xlarge TYPE=m5.2xlarge IDLE=1800 CONFIGS=file://experiments/aws/emr_configs.json \
#    BOOTSTRAP=s3://us-east-1-s3-gnn/LakeGRL/code/bootstrap.sh CODE=s3://us-east-1-s3-gnn/LakeGRL/code/emo-pipeline.tar.gz \
#    KEY=... SUBNET=... scripts/emr_cluster.sh create 4
#
#  Primary and workers are the same type, so baselines or Phase 1 on the primary run on worker-class hardware.
#  The cluster terminates itself after an hour idle, so a forgotten cluster costs at most an hour.
#  Then on the primary: git clone, bash scripts/emr_setup.sh, bash scripts/emr_icde.sh dryrun.
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail
TYPE=${TYPE:-r6i.2xlarge}            # 8 vCPUs (4 physical cores), 64 GiB
RELEASE=${RELEASE:-emr-7.5.0}        # Spark 3.5, Delta 3.2 (delta-spark 3.2.0 locally), Python 3.9
BUCKET=${BUCKET:-us-east-1-s3-gnn}
REGION=${REGION:-us-east-1}
MASTER_TYPE=${MASTER_TYPE:-$TYPE}
IDLE=${IDLE:-3600}                   # idle auto-termination, seconds
CONFIGS=${CONFIGS:-'[{"Classification":"delta-defaults","Properties":{"delta.enabled":"true"}},
                     {"Classification":"spark","Properties":{"maximizeResourceAllocation":"false"}}]'}

case "${1:-}" in
create)
    W=${2:-1}
    BA=(); [ -z "${BOOTSTRAP:-}" ] || BA=(--bootstrap-actions "Path=$BOOTSTRAP,Args=[${CODE:?CODE=<code tarball on S3>}]")
    # KEY (SSH) and SUBNET are optional: without them the default VPC, and the work runs as steps (below)
    EA=$(echo "${KEY:+KeyName=$KEY,}${SUBNET:+SubnetId=$SUBNET}" | sed 's/,$//')
    EC2=(); [ -z "$EA" ] || EC2=(--ec2-attributes "$EA")
    aws emr create-cluster --region "$REGION" --name "lakegrl-icde-$(date +%m%d-%H%M)" --release-label "$RELEASE" \
        --applications Name=Hadoop Name=Spark \
        --instance-groups "InstanceGroupType=MASTER,InstanceCount=1,InstanceType=$MASTER_TYPE" \
                          "InstanceGroupType=CORE,InstanceCount=$W,InstanceType=$TYPE" \
        --configurations "$CONFIGS" "${BA[@]}" "${EC2[@]}" \
        --use-default-roles --log-uri "s3://$BUCKET/emr-logs/" \
        --auto-termination-policy IdleTimeout=$IDLE --query ClusterId --output text
    ;;
step)
    # runs experiments/aws/run.sh <what> [arg] on the primary as an EMR step (no SSH); prints the step id
    C=${2:?cluster id}; WHAT=${3:?smoke|expA|expB-read|expB-train|nodes}; ARG=${4:+,$4}
    aws emr add-steps --region "$REGION" --cluster-id "$C" --query 'StepIds[0]' --output text --steps \
        "Type=CUSTOM_JAR,Name=lakegrl-$WHAT${4:-},ActionOnFailure=CONTINUE,Jar=command-runner.jar,Args=[bash,/opt/lakegrl/emo-pipeline/experiments/aws/run.sh,$WHAT$ARG]"
    ;;
status)
    C=${2:?cluster id}
    aws emr describe-cluster --region "$REGION" --cluster-id "$C" --query 'Cluster.Status.State' --output text
    aws emr list-steps --region "$REGION" --cluster-id "$C" --query 'Steps[].[Name,Status.State,Status.Timeline.StartDateTime]' --output text
    ;;
resize)
    C=${2:?cluster id}; W=${3:?workers}
    G=$(aws emr describe-cluster --region "$REGION" --cluster-id "$C" \
        --query "Cluster.InstanceGroups[?InstanceGroupType=='CORE'].Id" --output text)   # CLI v2 has no list-instance-groups
    aws emr modify-instance-groups --region "$REGION" --cluster-id "$C" \
        --instance-groups "InstanceGroupId=$G,InstanceCount=$W"
    echo "core group $G -> $W workers; wait until 'yarn node -list' on the primary shows $W RUNNING"
    ;;
terminate)
    aws emr terminate-clusters --region "$REGION" --cluster-ids "${2:?cluster id}"
    ;;
*)
    sed -n '3,12p' "$0"; exit 1 ;;
esac
