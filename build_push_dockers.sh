sh setup.sh
docker build --tag laituan245/kairos_coref-v2 .
docker push laituan245/kairos_coref-v2
docker tag laituan245/kairos_coref-v2 laituan245/kairos_coref-v2:api
docker push laituan245/kairos_coref-v2:api
