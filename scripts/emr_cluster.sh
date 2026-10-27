#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  Create, resize or terminate the ICDE EMR cluster from a machine with the AWS CLI. Creating starts billing:
#  only with Bilal's OK on the estimate (plan doc, "AWS batch").
#
#    KEY=<ec2 key pair> SUBNET=<subnet id> scripts/emr_cluster.sh create [workers]   -> prints the cluster id
#    scripts/emr_cluster.sh resize <cluster id> <workers>
#    scripts/emr_cluster.sh terminate <cluster id>
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

case "${1:-}" in
create)
    W=${2:-1}
    aws emr create-cluster --region "$REGION" --name "lakegrl-icde-$(date +%m%d-%H%M)" --release-label "$RELEASE" \
        --applications Name=Hadoop Name=Spark \
        --instance-groups "InstanceGroupType=MASTER,InstanceCount=1,InstanceType=$TYPE" \
                          "InstanceGroupType=CORE,InstanceCount=$W,InstanceType=$TYPE" \
        --configurations '[{"Classification":"delta-defaults","Properties":{"delta.enabled":"true"}},
                           {"Classification":"spark","Properties":{"maximizeResourceAllocation":"false"}}]' \
        --ec2-attributes "KeyName=${KEY:?KEY=<ec2 key pair>},SubnetId=${SUBNET:?SUBNET=<subnet id>}" \
        --use-default-roles --log-uri "s3://$BUCKET/emr-logs/" \
        --auto-termination-policy IdleTimeout=3600 --query ClusterId --output text
    ;;
resize)
    C=${2:?cluster id}; W=${3:?workers}
    G=$(aws emr list-instance-groups --region "$REGION" --cluster-id "$C" \
        --query "InstanceGroups[?InstanceGroupType=='CORE'].Id" --output text)
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
