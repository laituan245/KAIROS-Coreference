sh setup.sh
docker build --tag laituan245/kairos_coref .
docker push laituan245/kairos_coref
docker tag laituan245/kairos_coref laituan245/kairos_coref:no_exit
docker push laituan245/kairos_coref:no_exit
docker tag laituan245/kairos_coref laituan245/kairos_coref:0.1
docker push laituan245/kairos_coref:0.1
