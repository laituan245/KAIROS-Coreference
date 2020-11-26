sh setup.sh
docker build --tag laituan245/kairos_coref .
docker push laituan245/kairos_coref
docker tag laituan245/kairos_coref laituan245/kairos_coref:api
docker push laituan245/kairos_coref:api
docker tag laituan245/kairos_coref laituan245/kairos_coref:dev
docker push laituan245/kairos_coref:dev
