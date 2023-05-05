epoch=500
beta=0.9
local_step=1

n_nodes=25
batch=32

optimizer=gossip
dataset=cifar100

config=./config/${n_nodes}_node.json

for seed in {0,} ; do
    for lr in {0.1,0.01,0.001} ; do
	for alpha in {0.1,} ; do
	    for nw in {one_peer_base,two_peer_base,one_peer_exp} ; do
		log_path=results_final/${dataset}_vgg/${optimizer}/node_${n_nodes}/${local_step}_local_step/${seed}/alpha_${alpha}/${nw}/lr_${lr}_beta_${beta}/
		mkdir -p ${log_path}
		
		python evaluate.py ${log_path} --model vgg --optimizer ${optimizer} --dataset ${dataset} --seed ${seed} --config ${config} --node_list 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 --nw ${nw} --lr ${lr} --epoch ${epoch} --alpha ${alpha} --beta ${beta} --local_step ${local_step} --batch ${batch}
		
	    done
	done
    done
done
